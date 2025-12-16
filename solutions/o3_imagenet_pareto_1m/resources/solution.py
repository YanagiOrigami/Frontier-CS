import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
import numpy as np

class _MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=None, dropout_p: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [896, 512, 256]  # chosen to stay under 1M params with BN
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # ----------------- setup -----------------
        if metadata is None:
            metadata = {}
        device = metadata.get("device", "cpu")
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 1_000_000)

        self._set_seeds(42)

        # ----------------- build model -----------------
        model = _MLP(input_dim, num_classes)
        if self._param_count(model) > param_limit:
            # fallback to a smaller config if needed
            small_hidden = [768, 512, 256]
            model = _MLP(input_dim, num_classes, hidden_dims=small_hidden)
        assert self._param_count(model) <= param_limit
        model.to(device)

        # ----------------- training setup -----------------
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        max_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        patience, patience_counter = 30, 0

        # ----------------- training loop -----------------
        for epoch in range(max_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_acc = self._evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > patience:
                break

        # ----------------- return best model -----------------
        model.load_state_dict(best_state)
        model.to(device)
        return model

    @staticmethod
    def _evaluate(model: nn.Module, loader, device: str) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        return correct / total if total else 0.0

    @staticmethod
    def _param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _set_seeds(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
