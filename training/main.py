import os
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import load_dataset

from transformers import DistilBertTokenizerFast

from model import SentencesClassification


def preprocess(dataset, tokenizer, batch_size):
    def encode(data):
        return tokenizer(
            data["hypothesis"], data["premise"], truncation=True, padding="max_length"
        )

    dataset_pre_processed = dataset.map(
        encode, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    dataset_pre_processed.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )  # The model needs tensor as inputs
    dataset_pre_processed = dataset_pre_processed.rename_column(
        "label", "labels"
    )  # The model takes as arg labels not label

    return (
        dataset_pre_processed["train"],
        dataset_pre_processed["validation"],
        dataset_pre_processed["test"],
    )


if __name__ == "__main__":

    configs = {"BATCH_SIZE": 32, "NUM_LABEL": 3, "NUM_EPOCH": 10}

    snli = load_dataset("snli")
    snli = snli.filter(lambda example: example["label"] != -1)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    train_data, validation_data, test_data = preprocess(
        dataset=snli, tokenizer=tokenizer, batch_size=configs["BATCH_SIZE"]
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=configs["BATCH_SIZE"],
        shuffle=False,
        num_workers=3,
    )
    validation_data_loader = torch.utils.data.DataLoader(
        dataset=validation_data,
        batch_size=configs["BATCH_SIZE"],
        shuffle=False,
        num_workers=3,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=configs["BATCH_SIZE"],
        shuffle=False,
        num_workers=3,
    )
    model = SentencesClassification(params=configs)

    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=["val_accuracy"],
        dirpath="models/",
        filename="distilbert-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=3,
        mode="max",
    )

    # training
    trainer = pl.Trainer(
        gpus=-1, max_epochs=10, logger=tb_logger
    )  # , limit_train_batches=1)
    trainer.fit(model, train_data_loader, validation_data_loader)
    trainer.test()
