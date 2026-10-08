"""EquiMod (SimCLR base) ResNet-18 on CIFAR-10.

Follows the official CIFAR-10 recipe of Devillers & Lefort (ICLR 2023): 800
epochs, batch 512, LARS lr 4.0, invariance temperature 0.5, equivariance
temperature 0.2, lambda 1. The paper reports 92.79% linear top-1 against 90.96%
for its SimCLR baseline, so the claim is the gap between the two. Accuracy here
is the library's online linear probe, not the paper's 90-epoch offline linear
evaluation, so only that gap is comparable, not the absolute numbers.

``METHOD=simclr`` trains the library's SimCLR with the same optimiser, schedule
and data pipeline (the un-augmented view is loaded but unused). Its backbone
sees each view in a separate pass, whereas EquiMod runs one pass over
[original, view1, view2] as in the official code, so batch-norm statistics
differ between the two runs. Epoch count is read from the ``MAX_EPOCHS`` env
var (default 800).
"""

import os
import sys
import types
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods import EquiMod, SimCLR

# Mean / std of the 15-d augmentation-parameter vector under the augmentation
# below, taken from the official implementation (src/main.py). Layout matches
# ``EquiMod.augmentation_params``.
PARAM_MEAN = [
    4.3122, 4.3216, 23.369, 23.374, 0.49998, 0.80087, 1.2025, 1.4007,
    1.5964, 1.8004, 0.99993, 1.0002, 0.99986, 2.2321e-07, 0.19965,
]  # fmt: skip
PARAM_STD = [
    3.9740, 3.9851, 4.9544, 4.9539, 0.5000, 0.3993, 1.1651, 1.0210,
    1.0200, 1.1669, 0.2066, 0.2068, 0.2066, 0.0517, 0.3997,
]  # fmt: skip


def _augmented_view():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(
            (32, 32), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


def _original_view():
    return transforms.Compose(
        transforms.RGB(),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


def equimod_forward(self, batch, stage):
    """Forward for EquiMod and its SimCLR baseline.

    Batch format:
        - Training: ``{"views": [original, view1, view2]}``; each augmented
          view carries the parameters its transforms sampled.
        - Eval: single dict with ``"image"`` key.

    Returns:
        Dictionary with ``"loss"`` (training only), ``"embedding"`` and
        ``"label"``.
    """
    if "image" in batch:
        output = SimCLR.forward(self, batch["image"])
        return {"embedding": output.embedding, "label": batch["label"].long()}

    original, view1, view2 = batch["views"]
    if isinstance(self, EquiMod):
        output = EquiMod.forward(
            self,
            view1["image"],
            view2["image"],
            original["image"],
            EquiMod.augmentation_params(view1),
            EquiMod.augmentation_params(view2),
        )
        self.log_dict(
            {
                f"{stage}/inv": output.invariance_loss,
                f"{stage}/equi": output.equivariance_loss,
                f"{stage}/equi_gain": output.equivariance_gain,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
    else:
        output = SimCLR.forward(self, view1["image"], view2["image"])

    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": output.loss,
        "embedding": output.embedding.detach(),
        "label": torch.cat([view1["label"], view2["label"]], dim=0).long(),
    }


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    method = os.environ.get("METHOD", "equimod")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 800))
    batch_size = 512
    num_workers = 10

    pl.seed_everything(0, workers=True)

    train_transform = transforms.MultiViewTransform(
        [_original_view(), _augmented_view(), _augmented_view()]
    )
    val_transform = _original_view()

    data_dir = str(get_data_dir("cifar10"))
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True),
                names=["image", "label"],
                transform=train_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True),
                names=["image", "label"],
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    if method == "equimod":
        module = EquiMod(
            encoder_name="resnet18",
            projector_dims=(2048, 2048, 128),
            temperature=0.5,
            equi_projector_dims=(2048, 2048, 128),
            equi_temperature=0.2,
            equi_weight=1.0,
            param_stats=(PARAM_MEAN, PARAM_STD),
            low_resolution=True,
        )
    elif method == "simclr":
        module = SimCLR(
            encoder_name="resnet18",
            projector_dims=(2048, 2048, 128),
            temperature=0.5,
            low_resolution=True,
        )
    else:
        raise ValueError(f"METHOD must be 'equimod' or 'simclr', got {method!r}")

    module.forward = types.MethodType(equimod_forward, module)
    module.optim = {
        "optimizer": {
            "type": "LARS",
            "lr": 4.0,
            "momentum": 0.9,
            "weight_decay": 1e-6,
            "eta": 1e-3,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": min(10, max_epochs / 2) / max_epochs,
            "start_factor": 1e-8,
            "end_lr": 1e-8,
            "total_steps": len(data.train) * max_epochs,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(module.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=20000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=module.embed_dim,
                k=10,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name=f"{method}-resnet18-cifar10",
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=1,
        accelerator="auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
