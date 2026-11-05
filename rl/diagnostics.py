"""Track the previously observed scale/concentration issues on a fixed probe."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.diagnostics import final_histograms, write_json


def capture(model, tokens, run_dir, iteration, seed):
    folder=run_dir/'diagnostics';folder.mkdir(exist_ok=True)
    probe_path=folder/'probe-tokens.pt'
    if not probe_path.exists():
        # Dedicated RNG: diagnostics must not change policy sampling/shuffling.
        ids=torch.randperm(len(tokens),generator=torch.Generator().manual_seed(seed))[:512]
        torch.save(tokens[ids].clone(),probe_path)
    probe=torch.load(probe_path,map_location='cpu',weights_only=True)
    data=TensorDataset(probe[:,None],torch.zeros(len(probe),1,dtype=torch.long))
    loader=DataLoader(data,batch_size=64,shuffle=False,generator=torch.Generator().manual_seed(seed))
    report=final_histograms(model,loader,next(model.parameters()).device,
                            max_observations=len(probe),max_values_per_layer=50000,seed=seed)
    report.update(model_state='ppo_update',update=iteration)
    report['activation_sample']['source']='fixed random sample from first captured PPO rollout; not held-out validation'
    relative=f'diagnostics/update-{iteration:06d}.json'
    write_json(run_dir/relative,report)
    features={}
    for name,stats in report['activation_stats'].items():
        if stats.get('module_type') not in ['ReLU','LeakyReLU']:continue
        mass=sorted([c['mean_absolute'] for c in stats.get('per_channel',[])],reverse=True)
        total=sum(mass)
        features[name]=dict(zero_fraction=stats['zero_fraction'],dormant_fraction=stats['dormant_fraction'],
                            top1_activity_share=mass[0]/total if total else None,
                            top4_activity_share=sum(mass[:4])/total if total else None)
    return dict(update=iteration,file=relative,probe_observations=len(probe),features=features)
