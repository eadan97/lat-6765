import glob
import os
from typing import List

import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import DeviceType
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics import KID, IS
from torchmetrics.image import PSNR, SSIM
from torchvision import transforms


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )


class LogImageGenerations(Callback):
    """Logs a validation batch and their generations to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            real_A, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )


class LogPairedMetrics(Callback):
    """Logs a validation batch and their generations to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self):
        super().__init__()
        self.ready = True
        self.val_ssim_a = SSIM(data_range=1.0, compute_on_step=False)
        self.val_ssim_b = SSIM(data_range=1.0, compute_on_step=False)
        self.test_ssim_a = SSIM(data_range=1.0, compute_on_step=False)
        self.test_ssim_b = SSIM(data_range=1.0, compute_on_step=False)
        self.val_psnr_a = PSNR(data_range=1.0, compute_on_step=False)
        self.val_psnr_b = PSNR(data_range=1.0, compute_on_step=False)
        self.test_psnr_a = PSNR(data_range=1.0, compute_on_step=False)
        self.test_psnr_b = PSNR(data_range=1.0, compute_on_step=False)

    def on_train_start(self, trainer, pl_module):
        if trainer._device_type == DeviceType.GPU:
            self.val_ssim_a = self.val_ssim_a.to(torch.device("cuda", 0))
            self.val_ssim_b = self.val_ssim_b.to(torch.device("cuda", 0))
            self.test_ssim_a = self.test_ssim_a.to(torch.device("cuda", 0))
            self.test_ssim_b = self.test_ssim_b.to(torch.device("cuda", 0))
            self.val_psnr_a = self.val_psnr_a.to(torch.device("cuda", 0))
            self.val_psnr_b = self.val_psnr_b.to(torch.device("cuda", 0))
            self.test_psnr_a = self.test_psnr_a.to(torch.device("cuda", 0))
            self.test_psnr_b = self.test_psnr_b.to(torch.device("cuda", 0))

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.ready:
            real_A, real_B, fake_A, fake_B = outputs["real_A"], outputs["real_B"], outputs["fake_A"], outputs["fake_B"]
            if pl_module.use_amp:
                fake_A = fake_A.float()
                fake_B = fake_B.float()

            real_A = real_A * 0.5 + 0.5
            real_B = real_B * 0.5 + 0.5
            fake_A = fake_A * 0.5 + 0.5
            fake_B = fake_B * 0.5 + 0.5

            # log val metrics
            self.val_ssim_a(fake_A, real_A)
            self.val_ssim_b(fake_B, real_B)
            self.val_psnr_a(fake_A, real_A)
            self.val_psnr_b(fake_B, real_B)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            ssim_a = self.val_ssim_a.compute()
            ssim_b = self.val_ssim_b.compute()
            psnr_a = self.val_psnr_a.compute()
            psnr_b = self.val_psnr_b.compute()
            log_dict = {"val/ssim": (ssim_a + ssim_b) / 2,
                        "val/psnr": (psnr_a + psnr_b) / 2,
                        "val/ssim_a": ssim_a,
                        "val/ssim_b": ssim_b,
                        "val/psnr_a": psnr_a,
                        "val/psnr_b": psnr_b}

            experiment.log(
                log_dict, step=pl_module.global_step
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.ready:
            real_A, real_B, fake_A, fake_B = outputs["real_A"], outputs["real_B"], outputs["fake_A"], outputs["fake_B"]
            if pl_module.use_amp:
                fake_A = fake_A.float()
                fake_B = fake_B.float()

            real_A = real_A * 0.5 + 0.5
            real_B = real_B * 0.5 + 0.5
            fake_A = fake_A * 0.5 + 0.5
            fake_B = fake_B * 0.5 + 0.5

            # log val metrics
            self.test_ssim_a(fake_A, real_A)
            self.test_ssim_b(fake_B, real_B)
            self.test_psnr_a(fake_A, real_A)
            self.test_psnr_b(fake_B, real_B)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            ssim_a = self.test_ssim_a.compute()
            ssim_b = self.test_ssim_b.compute()
            psnr_a = self.test_psnr_a.compute()
            psnr_b = self.test_psnr_b.compute()
            log_dict = {"test/ssim": (ssim_a + ssim_b) / 2,
                        "test/psnr": (psnr_a + psnr_b) / 2,
                        "test/ssim_a": ssim_a,
                        "test/ssim_b": ssim_b,
                        "test/psnr_a": psnr_a,
                        "test/psnr_b": psnr_b}

            experiment.log(
                log_dict, step=pl_module.global_step
            )


class LogUnpairedMetrics(Callback):
    """Logs a unpaired metrics for generative model.
    Metrics to log:
        - KID
        - IS

    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self,
                 kid_subset_size: int = 300,
                 is_splits: int = 10):
        super().__init__()
        self.ready = True
        self.val_kid_a = KID(subset_size=kid_subset_size)
        self.val_kid_b = KID(subset_size=kid_subset_size)
        self.test_kid_a = KID(subset_size=kid_subset_size)
        self.test_kid_b = KID(subset_size=kid_subset_size)
        self.val_is_a = IS(splits=is_splits)
        self.val_is_b = IS(splits=is_splits)
        self.test_is_a = IS(splits=is_splits)
        self.test_is_b = IS(splits=is_splits)

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: x * 127.5 + 127.5),
            transforms.Resize((299, 299), antialias=True)
        ])

    def on_train_start(self, trainer, pl_module):
        if trainer._device_type == DeviceType.GPU:
            self.val_kid_a = self.val_kid_a.to(torch.device("cuda", 0))
            self.val_kid_b = self.val_kid_b.to(torch.device("cuda", 0))
            self.test_kid_a = self.test_kid_a.to(torch.device("cuda", 0))
            self.test_kid_b = self.test_kid_b.to(torch.device("cuda", 0))
            self.val_is_a = self.val_is_a.to(torch.device("cuda", 0))
            self.val_is_b = self.val_is_b.to(torch.device("cuda", 0))
            self.test_is_a = self.test_is_a.to(torch.device("cuda", 0))
            self.test_is_b = self.test_is_b.to(torch.device("cuda", 0))

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.ready:
            real_A, real_B, fake_A, fake_B = outputs["real_A"], outputs["real_B"], outputs["fake_A"], outputs["fake_B"]
            if pl_module.use_amp:
                fake_A = fake_A.float()
                fake_B = fake_B.float()

            real_A = self.transforms(real_A).byte()
            real_B = self.transforms(real_B).byte()
            fake_A = self.transforms(fake_A).byte()
            fake_B = self.transforms(fake_B).byte()

            # log val metrics
            self.val_kid_a(real_A, real=True)
            self.val_kid_a(fake_A, real=False)
            self.val_kid_b(real_B, real=True)
            self.val_kid_b(fake_B, real=False)
            self.val_is_a(fake_A)
            self.val_is_b(fake_B)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            kid_a_mean, kid_a_std = self.val_kid_a.compute()
            kid_b_mean, kid_b_std = self.val_kid_b.compute()
            is_a_mean, is_a_std = self.val_is_a.compute()
            is_b_mean, is_b_std = self.val_is_b.compute()
            self.val_kid_a.reset()
            self.val_kid_b.reset()
            self.val_is_a.reset()
            self.val_is_b.reset()
            log_dict = {"val/kid": (kid_a_mean + kid_b_mean) / 2,
                        "val/is": (is_a_mean + is_b_mean) / 2,
                        "val/kid_a_mean": kid_a_mean,
                        "val/kid_a_std": kid_a_std,
                        "val/kid_b_mean": kid_b_mean,
                        "val/kid_b_std": kid_b_std,
                        "val/is_a_mean": is_a_mean,
                        "val/is_a_std": is_a_std,
                        "val/is_b_mean": is_b_mean,
                        "val/is_b_std": is_b_std,
                        }
            self.log_dict(log_dict)
            # experiment.log(
            #     log_dict, step=pl_module.global_step
            # )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.ready:
            real_A, real_B, fake_A, fake_B = outputs["real_A"], outputs["real_B"], outputs["fake_A"], outputs["fake_B"]
            if pl_module.use_amp:
                fake_A = fake_A.float()
                fake_B = fake_B.float()

            real_A = self.transforms(real_A).byte()
            real_B = self.transforms(real_B).byte()
            fake_A = self.transforms(fake_A).byte()
            fake_B = self.transforms(fake_B).byte()

            # log val metrics
            self.test_kid_a(real_A, real=True)
            self.test_kid_a(fake_A, real=False)
            self.test_kid_b(real_B, real=True)
            self.test_kid_b(fake_B, real=False)
            self.test_is_a(fake_A)
            self.test_is_b(fake_B)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            kid_a_mean, kid_a_std = self.test_kid_a.compute()
            kid_b_mean, kid_b_std = self.test_kid_b.compute()
            is_a_mean, is_a_std = self.test_is_a.compute()
            is_b_mean, is_b_std = self.test_is_b.compute()
            log_dict = {"test/kid": (kid_a_mean + kid_b_mean) / 2,
                        "test/is": (is_a_mean + is_b_mean) / 2,
                        "test/kid_a_mean": kid_a_mean,
                        "test/kid_a_std": kid_a_std,
                        "test/kid_b_mean": kid_b_mean,
                        "test/kid_b_std": kid_b_std,
                        "test/is_a_mean": is_a_mean,
                        "test/is_a_std": is_a_std,
                        "test/is_b_mean": is_b_mean,
                        "test/is_b_std": is_b_std,
                        }
            self.log_dict(log_dict)

            # experiment.log(
            #     log_dict, step=pl_module.global_step
            # )
