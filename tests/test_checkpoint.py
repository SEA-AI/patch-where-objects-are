"""Saving, resuming and warm-starting from checkpoints."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from patch_where_objects_are.config.detection import BackboneConfig, DetectionConfig
from patch_where_objects_are.models.factory import checkpoint_allow_missing_prefixes
from patch_where_objects_are.utils.checkpoint import (
    best_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from tests.helpers import tiny_detector


def _save(model, epoch, out, suffix="", optimizer=None, scaler=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters())
    scaler = scaler or torch.amp.GradScaler("cpu")
    save_checkpoint(
        model, optimizer, scaler, epoch, config={}, output_dir=str(out), suffix=suffix
    )


def test_resume_restores_model_optimizer_and_scaler_and_never_overwrites(tmp_path):
    model = nn.Linear(4, 4)
    # One AMP step: AdamW holds moments, the scaler a non-default scale.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
    scaler.scale(model(torch.ones(2, 4)).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    _save(model, epoch=2, out=tmp_path, suffix="_best")
    _save(model, 5, tmp_path, "_best", optimizer, scaler)
    saved = model.weight.detach().clone()
    nn.init.zeros_(model.weight)

    # Restore into a fresh optimizer and scaler.
    new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    new_scaler = torch.amp.GradScaler("cpu")
    ckpt = best_checkpoint(tmp_path)
    assert ckpt == tmp_path / "checkpoint_epoch0005_best.pth"
    start_epoch = load_checkpoint(model, new_optimizer, new_scaler, str(ckpt))

    assert start_epoch == 6
    torch.testing.assert_close(model.weight, saved)
    torch.testing.assert_close(
        new_optimizer.state_dict()["state"], optimizer.state_dict()["state"]
    )
    assert new_scaler.state_dict() == scaler.state_dict()
    with pytest.raises(FileExistsError):
        _save(model, epoch=5, out=tmp_path, suffix="_best")


def test_best_checkpoint_selects_by_kind(tmp_path):
    model = nn.Linear(4, 4)
    _save(model, epoch=3, out=tmp_path, suffix="_best_small")
    _save(model, epoch=4, out=tmp_path, suffix="_best")

    assert best_checkpoint(tmp_path) == tmp_path / "checkpoint_epoch0004_best.pth"
    assert (
        best_checkpoint(tmp_path, "best_small")
        == tmp_path / "checkpoint_epoch0003_best_small.pth"
    )


def test_mixed_resolution_model_warm_starts_from_uniform_checkpoint(tmp_path):
    uniform = tiny_detector("uniform")
    _save(uniform, epoch=34, out=tmp_path)
    mixed = tiny_detector("uniform_mixed")
    cfg = DetectionConfig(backbone=BackboneConfig(patch_selector="uniform_mixed"))

    # The patch-embedding fusion modules are new and keep their initialization.
    load_checkpoint(
        mixed,
        None,
        None,
        str(tmp_path / "checkpoint_epoch0034.pth"),
        allow_missing_prefixes=checkpoint_allow_missing_prefixes(cfg),
    )

    for name, value in uniform.state_dict().items():
        torch.testing.assert_close(mixed.state_dict()[name], value)
