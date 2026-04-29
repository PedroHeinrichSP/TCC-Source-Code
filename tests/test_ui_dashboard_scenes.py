import json
from pathlib import Path

from nvs_benchmark.ui.dashboard import _discover_scenes


def _write_minimal_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C6360F8CF0000020201004F94CEBE"
            "0000000049454E44AE426082"
        )
    )


def test_discover_scenes_supports_transforms_and_images_layout(tmp_path: Path) -> None:
    base_dir = tmp_path

    blender_scene = base_dir / "data" / "blender_synthetic" / "nerf_synthetic" / "lego"
    blender_scene.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(blender_scene / "frame_0000.png")
    (blender_scene / "transforms_train.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "./frame_0000.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    mip_scene_images = base_dir / "data" / "mipnerf360" / "garden" / "images"
    mip_scene_images.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(mip_scene_images / "0000.png")

    scenes = _discover_scenes(base_dir)
    assert scenes

    by_id = {item["id"]: item for item in scenes}

    blender_id = blender_scene.as_posix()
    assert blender_id in by_id
    blender_splits = {split["name"]: split for split in by_id[blender_id]["splits"]}
    assert "train" in blender_splits
    assert blender_splits["train"]["frames"]

    mip_scene_dir = mip_scene_images.parent
    mip_id = mip_scene_dir.as_posix()
    assert mip_id in by_id
    mip_splits = {split["name"]: split for split in by_id[mip_id]["splits"]}
    assert "images" in mip_splits
    assert mip_splits["images"]["frames"]
