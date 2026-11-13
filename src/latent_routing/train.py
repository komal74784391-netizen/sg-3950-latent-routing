import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import LatentRoutingModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = LatentRoutingModel(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_experts=model_cfg["num_experts"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    for epoch in range(config["train"]["epochs"]):
        model.train()
        running = 0.0
        for step, batch in enumerate(DataLoader(dataset=[], batch_size=config["train"]["batch_size"])):
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].to(device).view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}: mean loss {running:.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/latent_routing.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
