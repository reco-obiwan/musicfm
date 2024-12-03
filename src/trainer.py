from itertools import cycle

from torch import nn, einsum
from torch.optim import AdamW, lr_scheduler

from accelerate import Accelerator

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


class MusicFMTrainer(nn.Module):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        accelerate_kwargs,
        *,
        epoches=100,
        save_model_every=1,
        save_logs_every=1,
    ):
        super().__init__()

        self.epochs = epoches
        self.model = model

        self.accelerator = Accelerator(**accelerate_kwargs)
        self.device = self.accelerator.device

        # datasets
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # optimizers
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=1e-6,
        )

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.valid_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.valid_loader,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, wav in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            logits, _, losses, accuracies = self.model(wav)

            logger.info("logits: %s", logits["melspec_2048"].shape)
            logger.info("losses: %s", losses)
            logger.info("accuracies: %s", accuracies)

            # losses[""]
            # self.accelerator.backward(loss / self.grad_accum_every)
            # self.optimizer.step()
            # total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()

    def start_train(self):
        # 모델을 훈련 모드로 설정합니다
        self.model.train()

        self.steps = 0
        for epoch in range(self.epochs):
            self.accelerator.init_trackers(f"musicfm-v01-{epoch}")
            self.train_epoch()

            self.accelerator.end_training()
            val_loss, val_acc = self.validate()
            logger.info(f"Epoch {epoch+1}/{self.epochs}:")
            logger.info(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        return self

    def forward(self, x):
        raise Exception("Should not be called”")
