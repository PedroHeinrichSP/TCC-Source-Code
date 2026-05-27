from pathlib import Path

from scripts.download_tanks_and_temples import build_command, patch_official_downloader


def test_patch_official_downloader_and_command(tmp_path: Path) -> None:
    script_path = tmp_path / "download_t2_dataset.py"
    script_path.write_text(_sample_official_script(), encoding="utf-8")

    patch_official_downloader(script_path)

    patched = script_path.read_text(encoding="utf-8")
    command = build_command(script_path, Path("C:/datasets"), "intermediate", "image")

    assert command[-1] == "--calc_md5_off"
    assert 'with open(fname, encoding="utf-8-sig", errors="replace") as f:' in patched
    assert "def download_image_sets(pathname, scene, image_md5_dict, calc_md5):" in patched
    assert "def download_video(pathname, scene, image_md5_dict, calc_md5):" in patched


def _sample_official_script() -> str:
    return (
        "with open(fname) as f:\n"
        "    content = f.readlines()\n"
        "    content = [x.strip() for x in content]\n"
        "\n"
        "def download_image_sets(pathname, scene, image_md5_dict, calc_md5):\n"
        "    if (calc_md5):\n"
        "        h_md5 = generate_file_md5(download_file_local)\n"
        "        print('\\nmd5 downloaded: ' + h_md5)\n"
        "        print('md5 original:   ' + image_md5_dict[scene])\n"
        "        md5_check = h_md5 == image_md5_dict[scene]\n\n"
        "        if (md5_check):\n"
        "            if (unpack):\n"
        "                pass\n"
        "\n"
        "def download_video(pathname, scene, image_md5_dict, calc_md5):\n"
        "    if (calc_md5):\n"
        "        h_md5 = generate_file_md5(download_file_local)\n"
        "        print('\\nmd5 downloaded: ' + h_md5)\n"
        "        print('md5 original:   ' + video_md5_dict[scene])\n"
        "        md5_check = h_md5 == video_md5_dict[scene]\n\n"
        "        if (md5_check):\n"
        "            if (unpack):\n"
        "                pass\n"
        "\n"
        "def check_image_sets(pathname, scene, image_md5_dict):\n"
        "    if os.path.exists(download_file_local):\n"
        "        h_md5 = generate_file_md5(download_file_local)\n"
        "        md5_check = h_md5 == image_md5_dict[scene]\n"
        "        if (md5_check):\n"
        "            ret_str = 'X'\n"
        "        else:\n"
        "            ret_str = '?'\n"
        "\n"
        "def check_video(pathname, scene, video_md5_dict):\n"
        "    if os.path.exists(download_file_local):\n"
        "        h_md5 = generate_file_md5(download_file_local)\n"
        "        md5_check = h_md5 == video_md5_dict[scene]\n"
        "        if (md5_check):\n"
        "            ret_str = 'X'\n"
        "        else:\n"
        "            ret_str = '?'\n"
    )