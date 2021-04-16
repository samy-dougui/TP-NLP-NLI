import torch
import pytorch_lightning as pl

from sklearn.metrics import f1_score

from transformers import DistilBertForSequenceClassification


class SentencesClassification(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=params["NUM_LABEL"],
            output_attentions=True,
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        return [optimizer], [scheduler], ["val_loss"]

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
        predictions = torch.argmax(outputs.logits, dim=1)
        f1_acc = f1_score(
            inputs["labels"].detach().cpu(),
            predictions.detach().cpu(),
            average="weighted",
        )
        accuracy = (predictions == inputs["labels"]).float().sum()
        metrics = {
            "val_f1_acc": f1_acc,
            "val_loss": loss,
            "val_accuracy": accuracy / inputs["input_ids"].shape[0],
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {
            "test_f1_acc": metrics["val_f1_acc"],
            "test_acc": metrics["val_accuracy"],
            "test_loss": metrics["val_loss"],
            "batch_size": len(batch),
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_epoch_end(self, output_results):
        global_f1 = 0
        global_loss = 0
        for result in output_results:
            global_f1 = result["val_f1_acc"] * result["batch_size"]
            global_loss = result["test_loss"] * result["batch_size"]
            global_acc = result["test_acc"] * result["batch_size"]
        metrics = {
            "final_test_f1_acc": global_f1,
            "final_test_acc": global_acc,
            "final_test_loss": global_loss,
        }
        self.log_dict(metrics, prog_bar=True)

    def setup(self, stage="fit"):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
