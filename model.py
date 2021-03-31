import torch
import pytorch_lightning as pl

from sklearn.metrics import f1_score

from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup


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
        scheduler = get_linear_schedule_with_warmup(optimizer, 5, 2)
        return [optimizer], [scheduler]

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

        predictions = torch.argmax(outputs, dim=1)
        predictions.detach().cpu()
        f1_acc = f1_score(
            inputs["labels"].detach().cpu(),
            predictions.detach().cpu(),
            average="weighted",
        )
        metrics = {"val_f1_acc": f1_acc, "val_loss": loss}
        self.log(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {
            "test_f1_acc": metrics["val_f1_acc"],
            "test_loss": metrics["val_loss"],
            "batch_size": len(batch),
        }
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, output_results):
        global_f1 = 0
        global_loss = 0
        for result in output_results:
            global_f1 = result["val_f1_acc"] * result["batch_size"]
            global_loss = result["test_loss"] * result["batch_size"]
        metrics = {"final_test_f1_acc": global_f1, "final_test_loss": global_loss}
        self.log(metrics)

    def setup(self, stage="fit"):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
