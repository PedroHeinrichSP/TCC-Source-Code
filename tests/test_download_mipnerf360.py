from pathlib import Path

from scripts.download_mipnerf360 import _has_mipnerf360_scene_content, _resolve_archive_path


def _write_minimal_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C6360F8CF0000020201004F94CEBE"
            "0000000049454E44AE426082"
        )
    )


def test_has_mipnerf360_scene_content_accepts_extracted_scene(tmp_path: Path) -> None:
    root = tmp_path / "data" / "mipnerf360"
    image_dir = root / "garden" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(image_dir / "frame_000.png")

    assert _has_mipnerf360_scene_content(root)


def test_resolve_archive_path_prefers_existing_local_zip(tmp_path: Path) -> None:
    target_path = tmp_path / "data" / "mipnerf360"
    target_path.mkdir(parents=True, exist_ok=True)
    archive_path = target_path.parent / "360_v2.zip"
    archive_path.write_bytes(b"zip-placeholder")

    resolved = _resolve_archive_path(target_path, "360_v2.zip", "")

    assert resolved == archive_path.resolve()
