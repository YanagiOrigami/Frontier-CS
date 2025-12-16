import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
from typing import List, Tuple, Dict

class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 1152,
        n_hidden_layers: int = 4,
        dropout: float = 0.2,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ):
        super().__init__()
        dims: List[int] = [input_dim] + [hidden_dim] * n_hidden_layers + [num_classes]
        layers: List[nn.Module] = []
        norms: List[nn.Module] = []
        acts: List[nn.Module] = []
        drops: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            norms.append(nn.BatchNorm1d(dims[i + 1]))
            acts.append(nn.GELU())
            drops.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList(drops)
        self.output = nn.Linear(dims[-2], dims[-1])
        if mean is not None and std is not None:
            self.register_buffer("mu", mean.clone().view(1, -1))
            self.register_buffer("sigma", std.clone().view(1, -1))
        else:
            self.mu = None
            self.sigma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)
        for lin, bn, act, dr in zip(self.layers, self.norms, self.acts, self.drops):
            x = dr(act(bn(lin(x))))
        x = self.output(x)
        return x


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim: int = metadata.get("input_dim", 384)
        num_classes: int = metadata.get("num_classes", 128)
        param_limit: int = metadata.get("param_limit", 5_000_000)
        device_str: str = metadata.get("device", "cpu")
        device: torch.device = torch.device(device_str)

        torch.manual_seed(42)
        random.seed(42)

        def compute_mean_std(loader) -> Tuple[torch.Tensor, torch.Tensor]:
            cnt = 0
            mean = torch.zeros(input_dim, dtype=torch.float64)
            mean_sq = torch.zeros(input_dim, dtype=torch.float64)
            for feats, _ in loader:
                feats = feats.to(torch.float64)
                cnt += feats.size(0)
                mean += feats.sum(0)
                mean_sq += (feats ** 2).sum(0)
            mean /= cnt
            var = mean_sq / cnt - mean ** 2
            std = torch.sqrt(var + 1e-5)
            return mean.to(torch.float32), std.to(torch.float32)

        mean, std = compute_mean_std(train_loader)

        hidden_dim = 1152
        n_hidden_layers = 4
        dropout_rate = 0.2

        model = MLPNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout_rate,
            mean=mean,
            std=std,
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # Fallback to narrower network if constraint violated
            hidden_dim = 1024
            model = MLPNet(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                n_hidden_layers=5,
                dropout=dropout_rate,
                mean=mean,
                std=std,
            ).to(device)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert param_count <= param_limit, "Parameter limit exceeded"

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=False, min_lr=1e-5
        )

        def evaluate(net: nn.Module, loader) -> float:
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for feats, targets in loader:
                    feats, targets = feats.to(device), targets.to(device)
                    outputs = net(feats)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
            return correct / total if total > 0 else 0.0

        best_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
        patience = 25
        epochs_no_improve = 0
        max_epochs = 300

        for epoch in range(max_epochs):
            model.train()
            for feats, targets in train_loader:
                feats, targets = feats.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(feats)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            val_acc = evaluate(model, val_loader)
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            if best_acc >= 0.99:
                break

        model.load_state_dict(best_state)
        model.eval()
        return model
