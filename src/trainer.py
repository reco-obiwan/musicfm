import os
from datetime import datetime

import torch
from torch import nn, einsum
from torch.optim import AdamW, lr_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.utils import logging
import tqdm

from musicfm import MusicFM25Hz
from utilities import find_most_recent_file

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

os.environ["WANDB_API_KEY"] = "dcece3af87a9f23fc730de68bd25e40369f1e476"


class MusicFMTrainer(nn.Module):
    def __init__(
        self,
        workdir,
        version,
        train_loader,
        valid_loader,
        accelerate_kwargs,
        *,
        epoches=100000,
        save_interval=1,
        log_interval=100,
    ):
        super().__init__()

        self.workdir = workdir
        self.version = version
        self.formatted_date = datetime.now().strftime("%Y%m%d%H")

        self.epochs = epoches

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            **accelerate_kwargs, kwargs_handlers=[ddp_kwargs], log_with="wandb"
        )
        self.device = self.accelerator.device

        # datasets
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 가장 최근 파일 찾기
        model_base_path = os.path.join(self.workdir, "musicfm")
        self.model_path = os.path.join(
            model_base_path,
            find_most_recent_file(model_base_path) or "pretrained_msd.pt",
        )
        
        self.model = MusicFM25Hz(model_config=os.path.join(self.workdir, "res", "model_config.json"))
          
        # optimizers
        lr = 2e-5
        betas = (0.9, 0.99)
        eps = 1e-6
        weight_decay = 1e-6

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        
        self._load_model()
        
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

        self.log_interval = log_interval
        self.save_interval = save_interval

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0
        
        for batch_idx, wav in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            logits, _, losses, accuracies = self.model(wav)

            loss = losses["melspec_2048"]
            accuracy = accuracies["melspec_2048"]

            self.accelerator.backward(loss)
            self.optimizer.step()
            
            #self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process and batch_idx % self.log_interval == 0 and batch_idx > 0:
                logger.info("-------------------------------")
                logger.info("[%s][train] loss: %s", batch_idx, loss.item())
                logger.info("[%s][train] accuracy: %s", batch_idx, accuracy.item())
                logger.info("-------------------------------")

                self.accelerator.log({"loss": loss, "accuracy": accuracy})

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):

        self.model.eval()
        total_loss = 0
        correct = 0
    
        for batch_idx, wav in enumerate(self.valid_loader):
            with torch.no_grad():
                logits, _, losses, accuracies = self.model(wav)
            
            loss = losses["melspec_2048"]
            accuracy = accuracies["melspec_2048"]
            
            if self.accelerator.is_main_process:
                logger.debug("-------------------------------")
                logger.debug("[%s][valid] loss: %s", batch_idx, loss.item())
                logger.debug("[%s][valid] accuracy: %s", batch_idx, accuracy.item())
                logger.debug("-------------------------------")

            total_loss += loss.item()

        if self.accelerator.is_main_process:
            avg_loss =  total_loss / len(self.valid_loader)
            self.accelerator.log({"valid_loss": avg_loss})
            self.accelerator.log({"valid_total_loss": total_loss})
            
        return total_loss, avg_loss

    def start_train(self):
        # 모델을 훈련 모드로 설정합니다
        self.model.train()

        try:
            self.accelerator.init_trackers(
                project_name=self.version,
                init_kwargs={
                    "wandb": {
                        "entity": "dreamus-company",
                    }
                },
            )

            self.steps = 0
            for epoch in range(self.epochs):
                self._train_epoch(epoch)
                logger.info("[%s] validation", epoch)
                self._validate()

                if epoch % self.save_interval == 0:
                    self._save_model(epoch)
        finally:
            self.accelerator.end_training()

        return self

    def _save_model(self, epoch):
        if not self.accelerator.is_main_process:
            return 
        
        pkg = dict(
            state_dict=self.accelerator.get_state_dict(
                self.accelerator.unwrap_model(self.model)
            ),
            optim=self.optimizer.state_dict(),
        )

        path = os.path.join(
            self.workdir, "musicfm", f"{self.version}_{self.formatted_date}.pt"
        )
        
        torch.save(pkg, path)
        logger.info("[%s]save model: %s", epoch, path)

    def _load_model(self, load_optimizer=False):
        """ Accelerator 적용 전에 호출되어야 함 """
        
        logger.info("loaded model: %s", self.model_path)
        pkg = torch.load(self.model_path, weights_only=False)
        
        state_dict = pkg["state_dict"]
        optim = pkg["optim"]
            
        for k, v in state_dict.items():
            logger.debug(k)
        
        self.model.load_state_dict(state_dict, strict=True)
        
        if load_optimizer:
            self.optimizer.load_state_dict(optim)
            logger.info("loaded optimizer")

    def forward(self, x):
        raise Exception("Should not be called”")
