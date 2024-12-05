import os
import wandb

from torch import nn, einsum
from torch.optim import AdamW, lr_scheduler

from accelerate import Accelerator, DistributedDataParallelKwargs

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

os.environ["WANDB_API_KEY"] = "dcece3af87a9f23fc730de68bd25e40369f1e476"

class MusicFMTrainer(nn.Module):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        accelerate_kwargs,
        *,
        epoches=100000,
        save_interval=1,
        log_interval=1,00
    ):
        super().__init__()

        self.epochs = epoches
        self.model = model

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            **accelerate_kwargs, kwargs_handlers=[ddp_kwargs]
        )
        self.device = self.accelerator.device

        # datasets
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # optimizers
        lr=2e-5
        betas=(0.9, 0.99)
        eps=1e-6
        weight_decay=1e-6
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
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
        
        wandb.init(name="musicfm-v01", project="musicfm", entity="dreamus-company")
        wandb.watch(self.model, log_freq=100)
        self.log_interval = log_interval
        self.save_interval = save_interval

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, wav in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            logits, _, losses, accuracies = self.model(wav)
            
            loss = losses["melspec_2048"]
            accuracy = accuracies["melspec_2048"]
            
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.accelerator.wait_for_everyone()
            
            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    
                logger.info("-------------------------------")
                logger.info("[%s] loss: %s", batch_idx, loss.item())
                logger.info("[%s] accuracy: %s", batch_idx, accuracy.item())
                logger.info("-------------------------------")
                
                wandb.log({"loss": loss})
                wandb.log({"accuracy": accuracy})

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, wav in enumerate(self.train_loader):
                logits, _, losses, accuracies = self.model(wav)
                loss = losses["melspec_2048"]
                total_loss += loss.item()


        avg_loss = total_loss / len(self.valid_loader)
        wandb.log({"valid_loss": avg_loss})
        return avg_loss

    def start_train(self):
        # 모델을 훈련 모드로 설정합니다
        self.model.train()

        self.steps = 0
        for epoch in range(self.epochs):
            self.accelerator.init_trackers(f"musicfm-v01-{epoch}")
            self._train_epoch()
            self.accelerator.end_training()

            val_loss = self._validate()
            
            if epoch % self.save_interval == 0:
                self._save():

        return self
    
    def _save (self):
        pkg = dict(
            model=self.accelerator.get_state_dict(
                self.accelerator.unwrap_model(self.model)
            ),
            optim=self.optimizer.state_dict(),
        )
        
        torch.save(pkg, path)

        model_path = Path(path)
        pkg = torch.load(str(model_path), map_location="cpu")
        torch.save(
            obj=self.origin_model.audio_model,
            f="data01/pretrained_models/ast_model_v12.pt",
        )
    
    def _load(self):
        pass
    
    def forward(self, x):
        raise Exception("Should not be called”")
