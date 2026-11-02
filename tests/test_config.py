"""YAML configs: loading, command-line overrides and run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from patch_where_objects_are.config.detection import load_detection_config, run_dir
from patch_where_objects_are.config.loader import parse_overrides

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(autouse=True)
def roots(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("OUTPUT_ROOT", "/out")


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=str)
def test_config_loads(path):
    cfg = load_detection_config(str(path))
    assert cfg.data.dataset in {"sea", "lars", "modd2"}
    assert cfg.data.train_json.startswith("/data/")
    assert cfg.train.output_root == "/out"


@pytest.mark.parametrize(
    ("config", "argv", "expected"),
    [
        ("uniform/lars_832.yaml", ["--train.seed", "42"], "/out/uniform_lars_832/s42"),
        (
            "adaptive/lars.yaml",
            ["--backbone.patch_selector", "entropy", "--backbone.token_budget", "576"],
            "/out/adaptive_entropy_lars_ft1_n576/s0",
        ),
        (
            "adaptive/sea.yaml",
            ["--backbone.selector_pool", "mean", "--train.seed", "2"],
            "/out/adaptive_objectness_meanpool_sea_ft1_n784/s2",
        ),
    ],
)
def test_overrides_and_run_dir(config, argv, expected):
    cfg = load_detection_config(str(CONFIGS / config), parse_overrides(argv))
    assert run_dir(cfg) == expected


@pytest.mark.parametrize(
    "argv", [["--train.sed", "1"], ["--trian.seed", "1"], ["--train.seed.x", "1"]]
)
def test_unknown_key_raises(argv):
    with pytest.raises(ValueError, match="unknown config key"):
        load_detection_config(
            str(CONFIGS / "uniform/lars_832.yaml"), parse_overrides(argv)
        )


@pytest.mark.parametrize(
    "argv",
    [["--train.use_amp", "--train.seed", "1"], ["--train.use_amp"], ["1"]],
)
def test_flag_without_value_raises(argv):
    with pytest.raises(ValueError, match=r"needs a value|unexpected argument"):
        parse_overrides(argv)


def test_override_values_parse_as_yaml():
    assert parse_overrides(
        [
            "--train.use_amp", "true",
            "--backbone.token_budget", "null",
            "--train.base_lr", "5e-5",
            "--train.layer_lr_decay", "0.9",
            "--train.seed", "-1",
            "--head.feat_strides", "[16, 32]",
            "--train.experiment_name", "run_a",
        ]
    ) == {
        "train.use_amp": True,
        "backbone.token_budget": None,
        "train.base_lr": 5e-5,
        "train.layer_lr_decay": 0.9,
        "train.seed": -1,
        "head.feat_strides": [16, 32],
        "train.experiment_name": "run_a",
    }  # fmt: skip
