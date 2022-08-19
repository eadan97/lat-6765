import functools
import itertools
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import Identity
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection, IS, FID

from src.losses.hinge_loss import GANLoss
from src.losses.perceptual_loss import PerceptualLoss
from src.models.modules.patchgan import PatchGanDiscriminator
from src.models.modules.resnet_generator import ResnetGenerator
from src.utils.cyclegan_image_paired import LogImagesCycleGan
from src.utils.image_pool import ImagePool
from src.utils.model_utils import init_weights
from src.utils.utils import process_unpaired_metrics, add_spectral_norm


class CycleGanModel(LightningModule):
    def __init__(self,
                 ndf: int = 64,
                 ngf: int = 64,
                 discriminator_layers=3,
                 norm_layer: str = 'instance',
                 use_dropout: bool = False,
                 pool_size: int = 50,
                 gan_mode: str = 'least_square',
                 soft_noisy_labels: bool = False,
                 lambda_identity: float = 0.5,
                 lambda_A: float = 10.0,
                 lambda_B: float = 10.0,
                 lambda_perception: float = 0.1,
                 lr_g: float = 2e-4,
                 lr_d: float = 2e-4,
                 beta1: float = 0.5,
                 use_spectral: bool = False,
                 log_num_samples=8,
                 is_splits: int = 10
                 ):
        super(CycleGanModel, self).__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        norm_layer = self.get_norm_layer(norm_layer)

        # Define networks
        self.netD_A = PatchGanDiscriminator(input_nc=3, ndf=ndf, n_layers=discriminator_layers, norm_layer=norm_layer)
        self.netD_B = PatchGanDiscriminator(input_nc=3, ndf=ndf, n_layers=discriminator_layers, norm_layer=norm_layer)

        self.netG_A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                      n_blocks=9)
        self.netG_B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                      n_blocks=9)

        if use_spectral:
            self.netD_A.apply(add_spectral_norm)
            self.netD_B.apply(add_spectral_norm)
            self.netG_A.apply(add_spectral_norm)
            self.netG_B.apply(add_spectral_norm)

        # initialize networks with a normal dist with 0.02 std
        # todo: move this to network constructor
        init_weights(self.netD_A)
        init_weights(self.netD_B)
        init_weights(self.netG_A)
        init_weights(self.netG_B)

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        # loss functions
        self.criterionGAN = GANLoss(gan_mode)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        if lambda_perception > 0:
            self.criterionPerception = PerceptualLoss()

        # self.metrics = LogUnpairedMetrics(self, kid_subset_size, is_splits)
        metrics = MetricCollection([
            FID(),
            IS(splits=is_splits)
        ])
        self.val_metrics_A = metrics.clone(prefix='val/A_')
        self.val_metrics_B = metrics.clone(prefix='val/B_')
        self.test_metrics_A = metrics.clone(prefix='test/A_')
        self.test_metrics_B = metrics.clone(prefix='test/B_')
        self.log_images = LogImagesCycleGan(log_num_samples)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def get_norm_layer(norm_type):
        """Return a normalization layer

        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance | none

        For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
        For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
        """

        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            def norm_layer(x):
                return Identity()
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def generate_fake(self, real_A: torch.Tensor, real_B: torch.Tensor):
        fake_B = self.netG_A(real_A)
        fake_A = self.netG_B(real_B)
        return fake_A, fake_B,

    def generate_reconstructions(self, fake_A: torch.Tensor, fake_B: torch.Tensor):
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)
        return rec_A, rec_B

    def on_train_start(self) -> None:
        self.log_images.trainer = self.trainer
        self.log_images.ready = True
        # self.metrics.ready = True
        # self.metrics.use_amp = self.use_amp
        # if self.use_amp:
        #     self.metrics.move_to_gpu()

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        real_A, real_B = batch
        fake_A, fake_B = self.generate_fake(real_A, real_B)

        # training step for generators
        if optimizer_idx == 0:
            lambda_A = self.hparams.lambda_A
            lambda_B = self.hparams.lambda_B
            lambda_identity = self.hparams.lambda_identity
            rec_A, rec_B = self.generate_reconstructions(fake_A, fake_B)
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            if self.hparams.lambda_identity > 0:
                idt_A = self.netG_A(real_B)
                loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_identity
                idt_B = self.netG_B(real_A)
                loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_identity
            else:
                loss_idt_A = 0
                loss_idt_B = 0
            if self.hparams.lambda_perception > 0:
                self.criterionPerception.set_target_image(real_A)
                self.criterionPerception.set_source_image(fake_B)
                loss_perception_A = self.criterionPerception.get_feature_loss() * self.hparams.lambda_perception
                self.criterionPerception.set_target_image(real_B)
                self.criterionPerception.set_source_image(fake_A)
                loss_perception_B = self.criterionPerception.get_feature_loss() * self.hparams.lambda_perception
            else:
                loss_perception_A = 0
                loss_perception_B = 0


            # GAN loss D_A(G_A(A))
            loss_G_A = self.criterionGAN(self.netD_A(fake_B), True, False) * 2
            # GAN loss D_B(G_B(B))
            loss_G_B = self.criterionGAN(self.netD_B(fake_A), True, False) * 2
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
            # combined loss and calculate gradients
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_perception_A + loss_perception_B
            self.log_dict({"train/loss_G": loss_G, "train/loss_G_A": loss_G_A, "train/loss_G_B": loss_G_B,
                           "train/loss_cycle_A": loss_cycle_A,
                           "train/loss_cycle_B": loss_cycle_B,
                           "train/loss_idt_A": loss_idt_A,
                           "train/loss_idt_B": loss_idt_B,
                           "train/loss_perception_A": loss_perception_A,
                           "train/loss_perception_B": loss_perception_B,
                           })
            return loss_G

        # training step for discriminators
        # discriminator A
        elif optimizer_idx == 1:
            self.set_requires_grad([self.netD_A], True)
            fake_B = self.fake_B_pool.query(fake_B)
            pred_real_D_A = self.netD_A(real_B)
            loss_D_A_real = self.criterionGAN(pred_real_D_A, True, True)
            # Fake
            pred_fake_D_A = self.netD_A(fake_B.detach())
            loss_D_A_fake = self.criterionGAN(pred_fake_D_A, False, True)
            # Combined loss and calculate gradients
            loss_D_A = loss_D_A_real + loss_D_A_fake
            self.log_dict({"train/loss_D_A": loss_D_A, "train/loss_D_A_real": loss_D_A_real,
                           "train/loss_D_A_fake": loss_D_A_fake})
            return loss_D_A

        # discriminator B
        elif optimizer_idx == 2:
            self.set_requires_grad([self.netD_B], True)
            fake_A = self.fake_A_pool.query(fake_A)
            pred_real_D_B = self.netD_B(real_A)
            loss_D_B_real = self.criterionGAN(pred_real_D_B, True, True)
            # Fake
            pred_fake_D_B = self.netD_B(fake_A.detach())
            loss_D_B_fake = self.criterionGAN(pred_fake_D_B, False, True)
            # Combined loss and calculate gradients
            loss_D_B = loss_D_B_real + loss_D_B_fake
            self.log_dict({"train/loss_D_B": loss_D_B, "train/loss_D_B_real": loss_D_B_real,
                           "train/loss_D_B_fake": loss_D_B_fake})
            return loss_D_B

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        real_A, real_B = batch
        fake_A, fake_B = self.generate_fake(real_A, real_B)
        # return {"real_A": real_A, "real_B": real_B, "fake_A": fake_A, "fake_B": fake_B}
        images = {"real_A": real_A, "real_B": real_B, "fake_A": fake_A, "fake_B": fake_B}
        if not self.trainer.sanity_checking:
            self.val_metrics_A.update(imgs=(real_A * 127.5 + 127.5).byte(), real=True)
            self.val_metrics_A.update(imgs=(fake_A * 127.5 + 127.5).byte(), real=False)
            self.val_metrics_B.update(imgs=(real_B * 127.5 + 127.5).byte(), real=True)
            self.val_metrics_B.update(imgs=(fake_B * 127.5 + 127.5).byte(), real=False)
            self.log_images.add_images(images)

    def validation_epoch_end(self, outputs: List[Any]):
        if not self.trainer.sanity_checking:
            self.log_dict(process_unpaired_metrics(self.val_metrics_A.compute(), self.val_metrics_B.compute()))
            self.log_images.log_images(False)

    def test_step(self, batch: Any, batch_idx: int):
        real_A, real_B = batch
        fake_A, fake_B = self.generate_fake(real_A, real_B)
        images = {"real_A": real_A, "real_B": real_B, "fake_A": fake_A, "fake_B": fake_B}
        if not self.trainer.sanity_checking:
            self.test_metrics_A.update(imgs=(real_A * 127.5 + 127.5).byte(), real=True)
            self.test_metrics_A.update(imgs=(fake_A * 127.5 + 127.5).byte(), real=False)
            self.test_metrics_B.update(imgs=(real_B * 127.5 + 127.5).byte(), real=True)
            self.test_metrics_B.update(imgs=(fake_B * 127.5 + 127.5).byte(), real=False)
            self.log_images.add_images(images)

    def test_epoch_end(self, outputs: List[Any]):
        if not self.trainer.sanity_checking:
            self.log_dict(process_unpaired_metrics(self.test_metrics_A.compute(), self.test_metrics_B.compute()))
            self.log_images.log_images(True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                       lr=self.hparams.lr_g,
                                       betas=(self.hparams.beta1, 0.999))
        optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                         lr=self.hparams.lr_d,
                                         betas=(self.hparams.beta1, 0.999))
        optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                         lr=self.hparams.lr_d,
                                         betas=(self.hparams.beta1, 0.999))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.trainer.max_epochs) / float(self.trainer.max_epochs + 1)
            return lr_l

        scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda_rule)
        scheduler_D_A = LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
        scheduler_D_B = LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)
        return [optimizer_G, optimizer_D_A, optimizer_D_B], [scheduler_G, scheduler_D_A, scheduler_D_B]
