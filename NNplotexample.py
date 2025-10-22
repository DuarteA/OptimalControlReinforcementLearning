import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# Reproducibility
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Target function R -> R to regress
def target_fn(x):
    # example: smooth non-linear function
    return torch.sin(2.0 * math.pi * x) + 0.3 * x

# Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_sizes=[64,64], output_dim=1):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def make_dataset(n_samples=200, noise_std=0.1, domain=(-1.5,1.5), dtype=torch.float32, device='cpu'):
    xs = torch.from_numpy(np.random.uniform(domain[0], domain[1], size=(n_samples,1))).to(dtype)
    ys = target_fn(xs).to(dtype)
    ys += noise_std * torch.randn_like(ys)
    return TensorDataset(xs, ys)

def train(model, loader, val_loader=None, epochs=300, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        running /= len(loader.dataset)
        if ep % max(1, epochs//10) == 0 or ep==1:
            msg = f"Epoch {ep}/{epochs} train_loss={running:.6f}"
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                msg += f" val_loss={val_loss:.6f}"
            print(msg)
    return model

def plot_results(model, dataset, domain=(-1.5,1.5), device='cpu', save_path=None):
    model.to(device).eval()
    xs = np.linspace(domain[0], domain[1], 400).reshape(-1,1).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(xs).to(device)).cpu().numpy()
    # scatter dataset
    data_x = torch.stack([t[0] for t in dataset]).cpu().numpy()
    data_y = torch.stack([t[1] for t in dataset]).cpu().numpy()
    plt.figure(figsize=(7,4))
    plt.scatter(data_x, data_y, s=20, alpha=0.5, label='train (noisy)')
    plt.plot(xs, np.sin(2*np.pi*xs) + 0.3*xs, 'k--', label='ground truth')
    plt.plot(xs, preds, 'r', lw=2, label='model')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden', type=int, nargs='*', default=[64,64])
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--noise', type=float, default=0.08)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--device', type=str, default='cpu')  # or 'cuda' if available
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    seed_all(args.seed)
    ds = make_dataset(n_samples=args.samples, noise_std=args.noise)
    # simple split
    n_val = max(20, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = MLP(input_dim=1, hidden_sizes=args.hidden, output_dim=1)
    train(model, train_loader, val_loader=val_loader, epochs=args.epochs, lr=args.lr, device=args.device)
    
    plot_results(model, ds, device=args.device)

if __name__ == "__main__":
    main()