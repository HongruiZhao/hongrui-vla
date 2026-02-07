"""Train VLA on dataset of image, state, action, and text instruction"""

import argparse
import os
import yaml
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from models.vla_diffusion_policy import VLADiffusionPolicy
from tqdm import trange


class TrainingDataset(Dataset):
    def __init__(self, path, resize_to=64):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3), uint8
        if img.shape[0] != self.resize_to or img.shape[1] != self.resize_to:
            img = cv2.resize(img, (self.resize_to, self.resize_to))

        img = rearrange(img, 'h w c -> c h w')
        img = torch.from_numpy(img).float() / 255.0  # (3, H, W)
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        return img, state, action, text_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to the training config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.get("log_dir", "runs"))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    dataset = TrainingDataset(cfg["dataset_path"], resize_to=cfg["resize_to"])
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=cfg["d_model"],
        diffusion_T=cfg["diffusion_T"]
    ).to(device)

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    num_epochs = cfg["epochs"]

    for epoch in trange(num_epochs):
        model.train()
        total_loss = 0.0
        for img, state, action, text_ids in loader:
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            loss = model.loss(img, text_ids, state, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)

        avg_loss = total_loss / len(dataset)
        writer.add_scalar("Loss/train", avg_loss, epoch)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": cfg["d_model"],
            "diffusion_T": cfg["diffusion_T"],
        },
        cfg["save_path"],
    )
    print("Saved checkpoint:", cfg["save_path"])
    writer.close()


if __name__ == "__main__":
    main()