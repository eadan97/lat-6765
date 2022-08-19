import torch
import wandb

from src.callbacks.wandb_callbacks import get_wandb_logger


class LogImagesCycleGan:
    def __init__(self, num_images):
        super().__init__()
        self.ready=False
        self.num_images = num_images
        self.trainer = None
        self.real_A = torch.tensor([])
        self.real_B = torch.tensor([])
        self.fake_A = torch.tensor([])
        self.fake_B = torch.tensor([])

    def add_images(self, outputs):
        if len(self.real_A) < self.num_images and self.ready:
            real_A, real_B, fake_A, fake_B = outputs["real_A"], outputs["real_B"], outputs["fake_A"], outputs["fake_B"]

            self.real_A = torch.cat((self.real_A, real_A.cpu()))
            self.real_B = torch.cat((self.real_B, real_B.cpu()))
            self.fake_A = torch.cat((self.fake_A, fake_A.cpu()))
            self.fake_B = torch.cat((self.fake_B, fake_B.cpu()))

    def log_images(self, is_test=False):
        if self.ready:
            logger = get_wandb_logger(self.trainer)
            experiment = logger.experiment
            # log the images as wandb Image
            stage = 'test' if is_test else 'val'
            experiment.log(
                {
                    f"{stage}/images/real_A": [wandb.Image(x) for x in self.real_A[: self.num_images]],
                    f"{stage}/images/real_B": [wandb.Image(x) for x in self.real_B[: self.num_images]],
                    f"{stage}/images/fake_A": [wandb.Image(x) for x in self.fake_A[: self.num_images]],
                    f"{stage}/images/fake_B": [wandb.Image(x) for x in self.fake_B[: self.num_images]],
                    "global_step": self.trainer.global_step
                }
            )
            self.reset_images()

    def reset_images(self):
        self.real_A = torch.tensor([])
        self.real_B = torch.tensor([])
        self.fake_A = torch.tensor([])
        self.fake_B = torch.tensor([])
