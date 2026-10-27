#!/usr/bin/env python3
"""Train a feedforward or recurrent FOW policy on replay observations."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starters.neural.model import create_policy
from training.datasets import observation_planes, split_games


DATA_DIR = ROOT / "game_data"
MODEL_TYPE = "recurrent"  # "feedforward" or "recurrent"
OUTPUT = ROOT / "starters/neural/policy_recurrent.pt"
EPOCHS = 50
BATCH_SIZE = 256
SEQUENCE_BATCH_SIZE = 16  # Player-game sequences per recurrent batch.
SEQUENCE_LENGTH = 16  # Truncated backpropagation length, in observations.
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1
SEED = 7980
DEVICE = "auto"  # "auto", "cpu", "mps", or "cuda"


def run_epoch(model, loader, device, description, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = top5 = count = 0
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=description, unit="batch")
        for tokens, targets in progress:
            hidden = None  # New batch = new games; never share player memories.
            for start in range(0, tokens.shape[1], SEQUENCE_LENGTH):
                planes = observation_planes(tokens[:, start:start + SEQUENCE_LENGTH], device)
                labels = targets[:, start:start + SEQUENCE_LENGTH].to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits, hidden = model.forward_chunk(planes, hidden)
                valid = labels != -100  # Opponent turns and padding have no loss.
                scores, labels = logits[valid], labels[valid]
                if labels.numel():
                    loss = nn.functional.cross_entropy(scores, labels)
                    if training:
                        loss.backward()
                        model.clip_gradients()
                        optimizer.step()
                    count += labels.numel()
                    loss_sum += loss.item() * labels.numel()
                    correct += (scores.argmax(1) == labels).sum().item()
                    top5 += (scores.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
                    progress.set_postfix(loss=f"{loss_sum / count:.4f}")
                if hidden is not None:
                    hidden = hidden.detach()  # Keep memory, truncate gradients.
    return {"loss": loss_sum / count, "top1": correct / count, "top5": top5 / count}


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} already exists; change OUTPUT for a new run")
    torch.manual_seed(SEED)
    device = choose_device(DEVICE)
    if SEQUENCE_LENGTH < 1:
        raise ValueError("Sequence length must be positive")
    model = create_policy(MODEL_TYPE).to(device)
    train_paths, validation_paths = split_games(
        DATA_DIR.glob("*.json"), VALIDATION_FRACTION, SEED)
    loader_options = dict(batch_size=BATCH_SIZE, sequence_batch_size=SEQUENCE_BATCH_SIZE, seed=SEED)
    train_loader = model.make_loader(train_paths, shuffle=True, **loader_options)
    validation_loader = model.make_loader(validation_paths, **loader_options)
    train, validation = train_loader.dataset, validation_loader.dataset
    print(f"Device: {device}; model: {MODEL_TYPE}", flush=True)
    print(f"Train: {train.games:,} games / {train.positions:,} positions; "
          f"validation: {validation.games:,} games / {validation.positions:,} positions", flush=True)
    print("Accuracy is over all 4,096 classes, before inference-time legal-move filtering.", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device,
                                  f"Epoch {epoch}/{EPOCHS} train", optimizer)
        validation_metrics = run_epoch(model, validation_loader, device,
                                       f"Epoch {epoch}/{EPOCHS} eval")
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"top1 {train_metrics['top1']:.2%}, top5 {train_metrics['top5']:.2%} | "
            f"eval loss {validation_metrics['loss']:.4f}, "
            f"top1 {validation_metrics['top1']:.2%}, top5 {validation_metrics['top5']:.2%}",
            flush=True,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            checkpoint = {
                "kind": MODEL_TYPE,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(OUTPUT)
            print(f"Saved best policy to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
