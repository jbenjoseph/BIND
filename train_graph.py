import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dataset
from model_graph import ConditionModel


class MaskLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, yhat, y):
        mask = (y > -999).float()
        loss = self.loss(yhat.float(), y) * mask
        loss = torch.sum(loss)
        num_non_zero = torch.sum(mask)
        return (
            loss / num_non_zero
            if num_non_zero > 0
            else torch.tensor(0.0, device=y.device)
        )


class ConditionModelLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConditionModel()
        self.loss_fn = MaskLoss(nn.HuberLoss(delta=2.0, reduction="none"))
        self.mse_loss_fn = MaskLoss(nn.MSELoss(reduction="none"))
        self.loss_classifier = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-4
        self.iterations = 100000
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D"
        )
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model.eval()

    def forward(self, sequences, graphs):
        encoded_input = self.esm_tokenizer(
            sequences, padding="longest", return_tensors="pt"
        )
        esm_output = self.esm_model(
            **encoded_input.to(self.device), output_hidden_states=True
        )
        hidden_states = [x.detach().to(self.device) for x in esm_output.hidden_states]
        attention_mask = encoded_input["attention_mask"].to(self.device)
        return self.model(graphs, hidden_states, attention_mask)

    def training_step(self, batch, batch_idx):
        sequences = [item["sequence"] for item in batch]
        graphs = Batch.from_data_list([item["graph"] for item in batch]).to(self.device)
        y_ki = torch.tensor(
            [item["ki"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_ic50 = torch.tensor(
            [item["ic50"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_kd = torch.tensor(
            [item["kd"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_ec50 = torch.tensor(
            [item["ec50"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_classification = torch.tensor(
            [item["is_false_ligand"] for item in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yhat_ki, yhat_ic50, yhat_kd, yhat_ec50, yhat_classification = self(
                sequences, graphs
            )

        yhat_classification = torch.clamp(yhat_classification, min=-100, max=100)

        ki_loss = self.loss_fn(yhat_ki, y_ki)
        ic50_loss = self.loss_fn(yhat_ic50, y_ic50)
        kd_loss = self.loss_fn(yhat_kd, y_kd)
        ec50_loss = self.loss_fn(yhat_ec50, y_ec50)
        classification_loss = self.loss_classifier(
            yhat_classification, y_classification
        )

        total_loss = ki_loss + ic50_loss + kd_loss + ec50_loss + classification_loss

        self.log("Loss/train", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        sequences = [item["sequence"] for item in batch]
        graphs = Batch.from_data_list([item["graph"] for item in batch]).to(self.device)
        y_ki = torch.tensor(
            [item["ki"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_ic50 = torch.tensor(
            [item["ic50"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_kd = torch.tensor(
            [item["kd"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_ec50 = torch.tensor(
            [item["ec50"] for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        y_classification = torch.tensor(
            [item["is_false_ligand"] for item in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yhat_ki, yhat_ic50, yhat_kd, yhat_ec50, yhat_classification = self(
                sequences, graphs
            )

        yhat_classification = torch.clamp(yhat_classification, min=-100, max=100)

        ki_loss = self.loss_fn(yhat_ki, y_ki)
        ic50_loss = self.loss_fn(yhat_ic50, y_ic50)
        kd_loss = self.loss_fn(yhat_kd, y_kd)
        ec50_loss = self.loss_fn(yhat_ec50, y_ec50)
        classification_loss = self.loss_classifier(
            yhat_classification, y_classification
        )

        total_loss = ki_loss + ic50_loss + kd_loss + ec50_loss + classification_loss

        # Log validation loss
        self.log(
            "Loss/validation", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # Compute additional metrics
        yhat_bin_classification = (
            torch.sigmoid(yhat_classification).detach().cpu().numpy()
        )
        y_classification_np = y_classification.detach().cpu().numpy()

        ap = average_precision_score(
            y_classification_np.flatten(), yhat_bin_classification.flatten()
        )
        auroc = roc_auc_score(
            y_classification_np.flatten(), yhat_bin_classification.flatten()
        )
        accuracy = accuracy_score(
            y_classification_np.flatten(), np.round(yhat_bin_classification.flatten())
        )

        # Log metrics
        self.log("AUROC/validation", auroc, on_step=False, on_epoch=True)
        self.log("AP/validation", ap, on_step=False, on_epoch=True)
        self.log("Accuracy/validation", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iterations
        )
        return [optimizer], [scheduler]


batch_size = 16

train_sampler = None
val_sampler = None

if torch.cuda.device_count() > 1:
    train_sampler = DistributedSampler(dataset.train_dataset)
    val_sampler = DistributedSampler(dataset.validation_dataset)

train_loader = DataLoader(
    dataset.train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    collate_fn=lambda x: x,
)

val_loader = DataLoader(
    dataset.validation_dataset,
    batch_size=batch_size,
    sampler=val_sampler,
    num_workers=4,
    collate_fn=lambda x: x,
)

trainer = pl.Trainer(
    devices=4,
    accelerator="gpu",
    strategy="ddp",
    max_epochs=10,
    precision=16,
    log_every_n_steps=50,
    logger=pl.loggers.TensorBoardLogger("logs/"),
)

model = ConditionModelLightning()
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint("saves/model_final.ckpt")
