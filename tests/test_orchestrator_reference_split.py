from nvs_benchmark.core import DatasetSpec, HardwareProfile, RunConfig
from nvs_benchmark.core.orchestrator import Orchestrator


def test_resolve_reference_split_defaults_to_test_for_train_runs():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={},
    )

    assert Orchestrator._resolve_reference_split(config) == "test"


def test_resolve_reference_split_respects_explicit_override():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"reference_split": "val"},
    )

    assert Orchestrator._resolve_reference_split(config) == "val"


def test_resolve_reference_split_uses_method_eval_split_when_available():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"gs_eval_split": "test"},
    )

    assert Orchestrator._resolve_reference_split(config) == "test"


def test_resolve_reference_split_uses_dynamic_method_eval_split_when_available():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="d_nerf", root="/tmp/dataset", split="train"),
        method="gs_dynamic",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"gs_dynamic_eval_split": "test"},
    )

    assert Orchestrator._resolve_reference_split(config) == "test"
