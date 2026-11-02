"""Structural contract for all detection models."""

from __future__ import annotations

from torch import nn

# Coarse module groups used by training utilities (grad norms, freeze helpers, etc.).
DETECTION_MODULE_GROUPS: tuple[str, ...] = ("backbone", "fpn", "head")


def detection_parameter_group(param_name: str) -> str | None:
    """Return the coarse group for a parameter name, or None if ungrouped."""
    name = param_name.removeprefix("module.")
    for group in DETECTION_MODULE_GROUPS:
        if name == group or name.startswith(f"{group}."):
            return group
    return None


class DetectionModel(nn.Module):
    """Base class enforcing the backbone → fpn → head detector layout.

    All detectors in this codebase must expose exactly these three ``nn.Module``
    children so training utilities can apply layer-wise LR, gradient-norm logging,
    and backbone freeze/unfreeze consistently across backbone types.
    """

    backbone: nn.Module
    fpn: nn.Module
    head: nn.Module

    def __init__(
        self,
        *,
        backbone: nn.Module,
        fpn: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.validate_structure()

    def validate_structure(self) -> None:
        """Raise ``TypeError`` if this detector violates the shared layout."""
        errors: list[str] = []
        for name in DETECTION_MODULE_GROUPS:
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module):
                errors.append(
                    f"{name!r} must be an nn.Module, got {type(module).__name__}"
                )
        if errors:
            raise TypeError(
                f"{type(self).__name__} does not satisfy DetectionModel layout: "
                + "; ".join(errors)
            )
