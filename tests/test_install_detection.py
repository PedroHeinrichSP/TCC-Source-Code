from pathlib import Path

from nvs_benchmark.install import InstallCatalog, InstallItem, _is_installed, install_item_by_id


def _write_minimal_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C6360F8CF0000020201004F94CEBE"
            "0000000049454E44AE426082"
        )
    )


def test_is_installed_accepts_mipnerf360_scene_tree(tmp_path: Path) -> None:
    root = tmp_path / "data" / "mipnerf360"
    scene = root / "garden" / "images"
    scene.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(scene / "frame_000.png")

    item = InstallItem(
        item_id="mipnerf360",
        label="Mip-NeRF 360",
        path=str(root),
        url="",
        command="",
        size_mb=None,
    )

    assert _is_installed(item)


def test_is_installed_accepts_tanks_and_temples_scene_tree(tmp_path: Path) -> None:
    root = tmp_path / "data" / "tanks_and_temples"
    scene = root / "image_sets" / "Family"
    scene.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(scene / "frame_000.png")

    item = InstallItem(
        item_id="tanks_and_temples",
        label="Tanks and Temples",
        path=str(root),
        url="",
        command="",
        size_mb=None,
    )

    assert _is_installed(item)


def test_install_item_by_id_skips_existing_dataset_tree(tmp_path: Path) -> None:
    root = tmp_path / "data" / "mipnerf360"
    scene = root / "garden" / "images"
    scene.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(scene / "frame_000.png")

    item = InstallItem(
        item_id="mipnerf360",
        label="Mip-NeRF 360",
        path=str(root),
        url="https://example.invalid/mipnerf360.zip",
        command="python ./scripts/download_mipnerf360.py --pathname ./data/mipnerf360",
        size_mb=None,
    )
    catalog = InstallCatalog(datasets=[item], methods=[], notes=[])

    messages = install_item_by_id(catalog=catalog, item_id="mipnerf360", execute=False)

    assert messages == [f"[skip] {item.label}: already exists at {item.path}"]
