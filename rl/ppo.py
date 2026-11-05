"""Masked, feedforward PPO with complete-game GAE and fixed inference modes."""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn.utils import clip_grad_norm_

from rl.policy import masked_distribution
from training.datasets import observation_planes


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-3
    epochs: int = 4
    minibatch_size: int = 128
    clip_ratio: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5
    target_kl: float = .02
    temperature: float = .9


def optimizer_for(model, config):
    # The new critic can learn faster without applying that LR to the actor.
    return torch.optim.Adam([
        {'params': [p for name,p in model.named_parameters() if not name.startswith('value.')],
         'lr': config.learning_rate},
        {'params': list(model.value.parameters()), 'lr': config.value_learning_rate},
    ], eps=1e-5)


def evaluate_actions(model, tokens, masks, actions, temperature):
    model.eval()  # Fixed BN buffers/dropout in both rollout and optimization.
    device = next(model.parameters()).device
    logits, values, _ = model(observation_planes(tokens, device))
    dist = masked_distribution(logits, masks.to(device), temperature)
    return dist.log_prob(actions.to(device)), dist.entropy(), values[:, 0]


def explained_variance(predictions, targets):
    variance = targets.var(unbiased=False)
    return None if variance < 1e-12 else float(1-(targets-predictions).var(unbiased=False)/variance)


def value_quality(predictions, targets):
    predictions = torch.as_tensor(predictions, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=predictions.device)
    error = predictions-targets
    return dict(mse=float(error.square().mean()), rmse=float(error.square().mean().sqrt()),
                mae=float(error.abs().mean()), explained_variance=explained_variance(predictions, targets),
                prediction_mean=float(predictions.mean()), prediction_std=float(predictions.std(unbiased=False)),
                target_mean=float(targets.mean()), target_std=float(targets.std(unbiased=False)), samples=len(targets))


def update(model, optimizer, batch, config):
    if len(batch) < 2:
        raise ValueError('PPO requires at least two decisions to normalize advantages')
    model.eval()
    device = next(model.parameters()).device
    advantages = batch.advantages.to(device)
    advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
    returns = batch.returns.to(device)
    old = batch.old_log_probabilities.to(device)
    before = {name: p.detach().clone() for name,p in model.named_parameters()}
    stats = []
    gradients = {}
    stop = False
    for epoch in range(config.epochs):
        # This CPU RNG is saved in checkpoints for exact update-boundary resume.
        order = torch.randperm(len(batch))
        for start in range(0,len(batch),config.minibatch_size):
            ids = order[start:start+config.minibatch_size]
            logp, entropy, values = evaluate_actions(model, batch.tokens[ids], batch.masks[ids],
                                                     batch.actions[ids], config.temperature)
            log_ratio = logp-old[ids]
            ratio = log_ratio.exp()
            approximate_kl = ((ratio-1)-log_ratio).mean()
            if not torch.isfinite(approximate_kl):
                raise FloatingPointError('Nonfinite PPO probability ratio')
            if config.target_kl and approximate_kl > 1.5*config.target_kl:
                stop = True
                break
            surrogate = torch.minimum(ratio*advantages[ids],
                                      ratio.clamp(1-config.clip_ratio,1+config.clip_ratio)*advantages[ids])
            policy_loss = -surrogate.mean()
            value_loss = .5*(values-returns[ids]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coefficient*value_loss - config.entropy_coefficient*entropy_mean
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Record actual layer L2 before clipping, rather than infer it from LR.
            for name,module in model.named_modules():
                gs=[p.grad.detach().float().square().sum() for p in module.parameters(recurse=False)
                    if p.grad is not None]
                if gs:
                    gradients.setdefault(name,[]).append(float(torch.stack(gs).sum().sqrt()))
            norm = clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            stats.append(dict(total_loss=float(loss.detach()),policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                              entropy=float(entropy_mean.detach()),approximate_kl=float(approximate_kl.detach()),
                              ratio_clip_fraction=float(((ratio-1).abs()>config.clip_ratio).float().mean()),
                              pre_clip_gradient_l2=float(norm),samples=len(ids)))
        if stop:
            break
    if not stats:
        raise RuntimeError('PPO made no optimizer steps; check rollout/log-probability consistency')
    total = sum(row['samples'] for row in stats)
    report={key:sum(row[key]*row['samples'] for row in stats)/total
            for key in stats[0] if key != 'samples'}
    # Full-batch endpoint diagnostics avoid reporting only pre-update minibatch KL.
    logps, entropies, predictions = [], [], []
    with torch.no_grad():
        for start in range(0,len(batch),config.minibatch_size):
            end=start+config.minibatch_size
            p,e,v=evaluate_actions(model,batch.tokens[start:end],batch.masks[start:end],
                                   batch.actions[start:end],config.temperature)
            logps.append(p);entropies.append(e);predictions.append(v)
        new=torch.cat(logps);ratio=(new-old).exp();prediction=torch.cat(predictions)
        endpoint_kl=float(((ratio-1)-(new-old)).mean())
    if not math.isfinite(endpoint_kl):
        raise FloatingPointError('Nonfinite endpoint KL')
    layer_updates={}
    for name,module in model.named_modules():
        params=list(module.named_parameters(recurse=False))
        if not params:
            continue
        old_params=[before[f'{name}.{k}' if name else k] for k,p in params]
        delta=sum((p.detach()-q).float().square().sum() for (k,p),q in zip(params,old_params)).sqrt()
        scale=sum(q.float().square().sum() for q in old_params).sqrt()
        layer_updates[name]=dict(update_l2=float(delta),parameter_l2_before=float(scale),
                                 relative_l2=float(delta/scale) if scale>1e-12 else None)
    report.update(transitions=len(batch),optimizer_steps=len(stats),early_stop_kl=stop,
                  endpoint_kl=endpoint_kl,endpoint_entropy=float(torch.cat(entropies).mean()),
                  explained_variance_before=explained_variance(batch.old_values,batch.returns),
                  explained_variance_after=explained_variance(prediction,returns),
                  gradient_clipped_fraction=sum(s['pre_clip_gradient_l2']>config.max_grad_norm for s in stats)/len(stats),
                  gradient_global_max=max(s['pre_clip_gradient_l2'] for s in stats),
                  gradient_layer_l2={k:dict(mean=sum(v)/len(v),median=float(torch.tensor(v).median()),
                                            p95=float(torch.quantile(torch.tensor(v),.95)),max=max(v))
                                     for k,v in gradients.items()},
                  value_fit=dict(
                      gae_before=value_quality(batch.old_values, batch.returns),
                      gae_after=value_quality(prediction, returns),
                      monte_carlo_before=value_quality(batch.old_values, batch.discounted_returns),
                      monte_carlo_after=value_quality(prediction, batch.discounted_returns)),
                  advantage_mean=float(batch.advantages.mean()),
                  advantage_std=float(batch.advantages.std(unbiased=False)),
                  layer_updates=layer_updates,config=asdict(config))
    return report
