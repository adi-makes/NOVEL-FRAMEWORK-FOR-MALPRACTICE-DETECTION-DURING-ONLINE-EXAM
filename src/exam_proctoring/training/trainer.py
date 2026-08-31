from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.exam_proctoring.data.dataset import get_project_root

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        pos_weight: float = 3.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _move_batch(self, batch: Tuple[torch.Tensor, ...]):
        gaze, interaction, environment, labels = batch
        return (
            gaze.to(self.device),
            interaction.to(self.device),
            environment.to(self.device),
            labels.to(self.device),
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in loader:
            gaze, interaction, environment, labels = self._move_batch(batch)
            self.optimizer.zero_grad()
            logits = self.model(gaze, interaction, environment)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)

        return total_loss / max(1, count)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        all_logits = []
        all_labels = []

        for batch in loader:
            gaze, interaction, environment, labels = self._move_batch(batch)
            logits = self.model(gaze, interaction, environment)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)

        probs_np = 1.0 / (1.0 + np.exp(-logits_np))

        if len(np.unique(labels_np)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(labels_np, probs_np))

        return {
            "loss": total_loss / max(1, count),
            "auc": auc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 8,
        checkpoint_path: Optional[str] = None,
        model_name: str = "model",
        config_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_auc = val_metrics["auc"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                if checkpoint_path is not None:
                    cp = Path(checkpoint_path)
                    if not cp.is_absolute():
                        cp = get_project_root() / cp
                    cp.parent.mkdir(parents=True, exist_ok=True)

                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,
                            "val_metrics": val_metrics,
                            "history": history,
                            "model_name": model_name,
                            "config_meta": config_meta or {},
                        },
                        cp,
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
