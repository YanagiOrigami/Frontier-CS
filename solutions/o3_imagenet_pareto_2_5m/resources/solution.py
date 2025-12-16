import torch
import torch.nn as nn
from copy import deepcopy

class _MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1536, bias=True),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1536, 896, bias=True),
            nn.BatchNorm1d(896),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(896, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu") if metadata else "cpu"
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        model = _MLP(input_dim, num_classes).to(device)

        # Verify parameter budget
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > metadata["param_limit"]:
            raise RuntimeError(f"Parameter limit exceeded: {param_count} > {metadata['param_limit']}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

        best_model_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 30
        num_epochs = 200

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device).float()
                    targets = targets.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
            val_acc = correct / total if total > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break  # Early stopping

        model.load_state_dict(best_model_state)
        model.to("cpu")
        return model
