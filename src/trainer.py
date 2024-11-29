from torch import nn, einsum
from torch.optim import AdamW, lr_scheduler

from accelerate import Accelerator


class MusicFMTrainer(nn.Module):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        accelerate_kwargs,
        *,
        epoch_every_steps=10000,
        validation_check_every=10000,
        save_model_every=10000,
        save_logs_every=10000,
        num_train_steps=500000,
    ):
        super().__init__()

        self.num_train_steps = num_train_steps

        self.steps = 0
        self.model = model
        self.epoch_every_steps = epoch_every_steps

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

    def step(self):
        self.optimizer.zero_grad()

    def start(self):
        # 모델을 훈련 모드로 설정합니다
        self.model.train()

        self.steps = 0
        while self.steps < self.num_train_steps:
            if self.steps % self.epoch_every_steps == 0:
                # 만약 epoch_every_steps 이 10일 경우,
                # steps 0(epoch:1) -> 1 -> 2 -> ... -> 10(epoch:2) -> ...
                self.epoch = int(self.steps // self.epoch_every_steps) + 1
                self.accelerator.init_trackers(
                    f"musicfm-v01-{self.epoch}"
                )  # initiate위한 코드

            self.step()
            self.steps += 1

            if (self.steps % self.epoch_every_steps) == (self.epoch_every_steps - 1):
                # 만약 epoch_every_steps 이 10일 경우,
                # steps 0 -> 1 -> 2 -> ... -> 9(logging_end)
                self.accelerator.end_training()
        # steps는 현재 돌아가는 step, num_train_steps는 이때까지 돌아야한다는 것
        self.print("training complete")

        return self

    def forward(self, x):
        raise Exception("Should not be called”")
