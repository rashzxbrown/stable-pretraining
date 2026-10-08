"""EquiMod: an equivariance module on top of SimCLR.

Adds a second, *equivariant* latent space to SimCLR. An augmentation-conditioned
predictor maps the embedding of the un-augmented image to the embedding of each
augmented view, given the parameters of the augmentation that produced it. The
prediction is trained with a second NT-Xent loss, added to the usual invariance
loss, so the backbone keeps augmentation-related information that a purely
invariant objective would discard.

References:
    Devillers & Lefort. "EquiMod: An Equivariance Module to Improve
    Self-Supervised Learning." ICLR 2023. https://arxiv.org/abs/2211.01244

Example::

    from stable_pretraining.methods import EquiMod

    model = EquiMod(encoder_name="resnet18", low_resolution=True)

    # views come from a MultiViewTransform with three pipelines:
    # [un-augmented, augmented, augmented]
    original, view1, view2 = (v["image"] for v in batch["views"])
    params1 = EquiMod.augmentation_params(batch["views"][1])
    params2 = EquiMod.augmentation_params(batch["views"][2])

    out = model(view1, view2, original, params1, params2)
    out.loss.backward()
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.methods.simclr import SimCLR, SimCLROutput, _build_projector


_Stats = Union[torch.Tensor, Sequence[float]]


@dataclass
class EquiModOutput(SimCLROutput):
    """SimCLROutput extended with the two loss terms and an equivariance metric.

    Attributes:
        invariance_loss: NT-Xent loss between the two augmented views.
        equivariance_loss: NT-Xent loss between each view's equivariant
            embedding and its prediction from the un-augmented image.
        equivariance_gain: Mean of ``cos(y, y_hat) - cos(y, y_original)``.
            Positive when the predictor moves the un-augmented embedding
            towards the augmented one, i.e. when the augmentation is actually
            modelled.
    """

    invariance_loss: Optional[torch.Tensor] = None
    equivariance_loss: Optional[torch.Tensor] = None
    equivariance_gain: Optional[torch.Tensor] = None


class EquiMod(SimCLR):
    """EquiMod: SimCLR with an augmentation-conditioned equivariance module.

    Architecture (on top of the SimCLR backbone and invariance projector):
        - **Equivariance projector**: MLP mapping backbone features of the
          un-augmented image and of both augmented views to the equivariant
          space.
        - **Parameter projector**: standardises the augmentation-parameter
          vector and maps it to a small embedding.
        - **Predictor**: maps ``[equivariant embedding of the un-augmented
          image, parameter embedding]`` to the predicted equivariant embedding
          of the augmented view.
        - **Loss**: ``NT-Xent(z1, z2) + equi_weight * NT-Xent(y, y_hat)``.

    Note:
        Follows the official implementation where it differs from the paper:
        the equivariance loss is a standard NT-Xent over ``[y, y_hat]``, so
        the other view of the same image and the other predictions act as
        negatives (Eq. 4 of the paper excludes the sibling view). The
        projectors and the predictor reuse SimCLR's projector builder, whose
        final batch norm has a learnable scale (official: ``affine=False``).

    Args:
        encoder_name: timm model name or a pre-instantiated ``nn.Module``
            whose ``forward`` returns a ``[B, D]`` tensor.
        projector_dims: Hidden + output dimensions of the invariance projector.
        temperature: Temperature of the invariance NT-Xent loss.
        equi_projector_dims: Hidden + output dimensions of the equivariance
            projector.
        equi_temperature: Temperature of the equivariance NT-Xent loss.
        equi_weight: Weight ``lambda`` of the equivariance loss.
        param_dim: Size of the augmentation-parameter vector (15 for
            crop + flip + colour jitter + grayscale, 17 with Gaussian blur).
        param_projector_dims: Hidden + output dimensions of the parameter
            projector.
        predictor_hidden_dims: Hidden dimensions of the predictor. Empty gives
            the single linear layer used in the paper.
        param_stats: Optional ``(mean, std)`` of the parameter vector, each a
            tensor or sequence of length ``param_dim``, used to standardise
            it. If ``None``, a non-affine batch norm standardises it instead.
        low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
        pretrained: Load pretrained timm weights for the encoder.

    Example::

        model = EquiMod(encoder_name="resnet18", low_resolution=True)

        original = torch.randn(64, 3, 32, 32)
        v1, v2 = torch.randn(64, 3, 32, 32), torch.randn(64, 3, 32, 32)
        p1, p2 = torch.randn(64, 15), torch.randn(64, 15)
        out = model(v1, v2, original, p1, p2)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        features = model(v1).embedding  # [64, embed_dim]
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 128),
        temperature: float = 0.5,
        equi_projector_dims: Sequence[int] = (2048, 2048, 128),
        equi_temperature: float = 0.2,
        equi_weight: float = 1.0,
        param_dim: int = 15,
        param_projector_dims: Sequence[int] = (128,),
        predictor_hidden_dims: Sequence[int] = (),
        param_stats: Optional[Tuple[_Stats, _Stats]] = None,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__(
            encoder_name=encoder_name,
            projector_dims=projector_dims,
            temperature=temperature,
            low_resolution=low_resolution,
            pretrained=pretrained,
        )
        self.equi_weight = equi_weight
        self.param_dim = param_dim

        self.equi_projector = _build_projector(
            self.embed_dim, list(equi_projector_dims)
        )
        equi_dim = list(equi_projector_dims)[-1]
        self.param_projector = _build_projector(
            param_dim, list(param_projector_dims), final_bn_no_bias=False
        )
        self.predictor = _build_projector(
            equi_dim + list(param_projector_dims)[-1],
            [*predictor_hidden_dims, equi_dim],
        )
        self.equi_loss = NTXEntLoss(temperature=equi_temperature)

        if param_stats is None:
            self.param_norm = nn.BatchNorm1d(param_dim, affine=False)
        else:
            mean, std = (torch.as_tensor(s, dtype=torch.float) for s in param_stats)
            if mean.shape != (param_dim,) or std.shape != (param_dim,):
                raise ValueError(
                    f"param_stats must be two tensors of shape [{param_dim}], "
                    f"got {tuple(mean.shape)} and {tuple(std.shape)}"
                )
            if not (std > 0).all():
                raise ValueError(
                    "param_stats std must be strictly positive; use 1.0 for "
                    "parameters that are constant under your augmentations"
                )
            self.param_norm = None
            self.register_buffer("param_mean", mean)
            self.register_buffer("param_std", std)

    @staticmethod
    def augmentation_params(
        view: Mapping[str, torch.Tensor],
        crop: Optional[str] = "RandomResizedCrop",
        flip: Optional[str] = "RandomHorizontalFlip",
        jitter: Optional[str] = "ColorJitter",
        grayscale: Optional[str] = "RandomGrayscale",
        blur: Optional[str] = None,
    ) -> torch.Tensor:
        """Assemble the augmentation-parameter vector of one collated view.

        The transforms in ``stable_pretraining.data.transforms`` record what
        they sampled under their class name in the sample dict. This gathers
        those records into the layout of the official implementation:

        - ``[0:4]`` crop: top, left, height, width
        - ``[4]`` flip: 1 if flipped
        - ``[5]`` colour jitter applied, ``[6:10]`` order of its four
          operations, ``[10:14]`` brightness, contrast, saturation, hue factors
        - ``[14]`` grayscale: 1 if applied
        - ``[15:17]`` blur applied, sigma (only when ``blur`` is given)

        A skipped colour jitter is recorded as zeros by the transform; it is
        re-encoded here as the identity (order ``0,1,2,3``, factors
        ``1,1,1,0``) so that "not applied" and "applied with neutral factors"
        are close in parameter space.

        Args:
            view: One entry of ``batch["views"]`` after collation.
            crop: Key of the ``RandomResizedCrop`` record, or ``None`` to skip.
            flip: Key of the ``RandomHorizontalFlip`` record, or ``None``.
            jitter: Key of the ``ColorJitter`` record, or ``None``.
            grayscale: Key of the ``RandomGrayscale`` record, or ``None``.
            blur: Key of the ``PILGaussianBlur`` or ``GaussianBlur`` record,
                or ``None``.

        Returns:
            Float tensor of shape ``[B, P]``.
        """
        parts = []
        if crop is not None:
            parts.append(view[crop].float())
        if flip is not None:
            parts.append(view[flip].float().unsqueeze(1))
        if jitter is not None:
            record = view[jitter].float()
            factors, order = record[:, :4], record[:, 4:]
            # ``order`` is a permutation of 0..3 when applied, all zeros when skipped
            applied = order.sum(dim=1, keepdim=True) > 0
            factors = torch.where(applied, factors, factors.new_tensor([1, 1, 1, 0]))
            order = torch.where(applied, order, order.new_tensor([0, 1, 2, 3]))
            parts += [applied.float(), order, factors]
        if grayscale is not None:
            parts.append(view[grayscale].float().unsqueeze(1))
        if blur is not None:
            # GaussianBlur records (sigma_x, sigma_y), always equal
            record = view[blur].float()
            sigma = record.reshape(record.shape[0], -1)[:, :1]
            parts += [(sigma > 0).float(), sigma]
        return torch.cat(parts, dim=1)

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"expected augmentation parameters of size {self.param_dim} "
                f"(param_dim), got {params.shape[-1]}"
            )
        params = params.to(self.param_projector[0].weight.dtype)
        if self.param_norm is not None:
            return self.param_norm(params)
        return (params - self.param_mean) / self.param_std

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
        original: Optional[torch.Tensor] = None,
        params1: Optional[torch.Tensor] = None,
        params2: Optional[torch.Tensor] = None,
    ) -> EquiModOutput:
        """Forward pass.

        Args:
            view1: First augmented view [B, C, H, W] (or single view at eval).
            view2: Second augmented view [B, C, H, W]. If ``None``, returns
                only the backbone embedding (eval mode).
            original: Un-augmented image [B, C, H, W].
            params1: Augmentation parameters of ``view1`` [B, param_dim].
            params2: Augmentation parameters of ``view2`` [B, param_dim].

        Returns:
            :class:`EquiModOutput`. ``embedding`` holds the backbone features
            of the two augmented views, [2B, D].
        """
        if view2 is None:
            embedding = self.backbone(view1)
            return EquiModOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )
        if original is None or params1 is None or params2 is None:
            raise ValueError(
                "EquiMod needs the un-augmented image and the augmentation "
                "parameters of both views: forward(view1, view2, original, "
                "params1, params2)"
            )

        B = view1.shape[0]
        # One pass over [original, view1, view2] so batch-norm statistics are
        # shared by the three, as in the official implementation.
        h = self.backbone(torch.cat([original, view1, view2], dim=0))

        z = self.projector(h[B:])
        invariance_loss = self.simclr_loss(z[:B], z[B:])

        y = self.equi_projector(h)
        y_original = y[:B].repeat(2, 1)
        y_views = y[B:]
        p = self.param_projector(
            self._normalize_params(torch.cat([params1, params2], dim=0))
        )
        y_hat = self.predictor(torch.cat([y_original, p], dim=1))
        equivariance_loss = self.equi_loss(y_views, y_hat)

        with torch.no_grad():
            gain = (
                F.cosine_similarity(y_views, y_hat)
                - F.cosine_similarity(y_views, y_original)
            ).mean()

        return EquiModOutput(
            loss=invariance_loss + self.equi_weight * equivariance_loss,
            embedding=h[B:],
            projection=z,
            invariance_loss=invariance_loss.detach(),
            equivariance_loss=equivariance_loss.detach(),
            equivariance_gain=gain,
        )
