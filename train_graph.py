import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from esm.inverse_folding.util import extract_coords_from_structure
from esm import pretrained

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
        self.esmfold_model = pretrained.esmfold_v1().eval().to(self.device)
        self.inverse_model, self.inverse_alphabet = (
            pretrained.esm_if1_gvp4_t16_142M_UR50()
        )
        self.inverse_model = self.inverse_model.eval().to(self.device)
        self.inverse_batch_converter = self.inverse_alphabet.get_batch_converter()

    def forward(self, sequences, graphs):
        # Generate structures using ESMFold
        structures = []
        for seq in sequences:
            with torch.no_grad():
                pdb_str = self.esmfold_model.infer_pdb(seq)
                structures.append(pdb_str)
        # Parse the PDB strings to extract coordinates
        coords_list = []
        seqs = []
        for pdb_str in structures:
            coords, seq = extract_coords_from_structure(pdb_str)
            coords_list.append(coords)
            seqs.append(seq)

        # Prepare data for the inverse folding model
        batch = []
        for seq, coords in zip(seqs, coords_list):
            batch.append((seq, coords))

        # Use the inverse folding model's batch converter
        (
            batch_labels,
            batch_strs,
            batch_coords,
            batch_tokens,
            padding_mask,
        ) = self.inverse_batch_converter(batch)

        batch_coords = batch_coords.to(self.device)
        batch_tokens = batch_tokens.to(self.device)
        padding_mask = padding_mask.to(self.device)

        with torch.no_grad():
            out = self.inverse_model(
                coords=batch_coords,
                tokens=batch_tokens,
                padding_mask=padding_mask,
                repr_layers=[16],
                return_contacts=False,
            )
            token_embeddings = out["representations"][16]

        # Use the padding_mask as attention_mask
        attention_mask = ~padding_mask

        # Pass token_embeddings and attention_mask to self.model
        return self.model(graphs, [token_embeddings], attention_mask)

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
