import torch
import pytorch_lightning as pl

from transformers import DistilBertForSequenceClassification


class SentencesClassification(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=params["NUM_LABEL"]
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = {
            "input_ids": train_batch["input_ids"].to(self.device),
            "labels": train_batch["labels"].to(self.device),
            "attention_mask": train_batch["attention_mask"].to(self.device),
        }
        outputs = self.model(
            inputs["input_ids"], inputs["attention_mask"], labels=inputs["labels"]
        )
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = {
            "input_ids": val_batch["input_ids"].to(self.device),
            "labels": val_batch["labels"].to(self.device),
            "attention_mask": val_batch["attention_mask"].to(self.device),
        }
        outputs = self.model(
            inputs["input_ids"], inputs["attention_mask"], labels=inputs["labels"]
        )
        loss = outputs.loss
        self.log("val_loss", loss)
