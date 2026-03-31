import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nvs_benchmark.ui.preview import _build_metrics_markdown, _scene_payload


class UiPreviewTests(unittest.TestCase):
    def test_scene_payload_fallback(self) -> None:
        payload = _scene_payload("./path/does/not/exist.json")
        self.assertTrue(payload["used_fallback"])
        self.assertGreater(payload["camera_count"], 0)
        self.assertGreater(len(payload["matrices"]), 0)

    def test_scene_payload_with_transform_matrix(self) -> None:
        with TemporaryDirectory() as tmp:
            scene_file = Path(tmp) / "transforms.json"
            image_file = Path(tmp) / "frame_0000.png"
            image_file.write_bytes(
                bytes.fromhex(
                    "89504E470D0A1A0A"
                    "0000000D4948445200000001000000010802000000907753DE"
                    "0000000C49444154789C6360F8CF0000020201004F94CEBE"
                    "0000000049454E44AE426082"
                )
            )
            scene_file.write_text(
                json.dumps(
                    {
                        "camera_angle_x": 0.691,
                        "frames": [
                            {
                                "file_path": "./frame_0000.png",
                                "transform_matrix": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.5],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = _scene_payload(str(scene_file))
            self.assertFalse(payload["used_fallback"])
            self.assertEqual(payload["camera_count"], 1)
            self.assertIn("frame_images", payload)
            self.assertEqual(len(payload["frame_images"]), 1)
            self.assertIsNotNone(payload["frame_images"][0])

    def test_metrics_markdown_contains_headers(self) -> None:
        table = _build_metrics_markdown(["nerf_static"], {})
        self.assertIn("| Metodo |", table)
        self.assertIn("PSNR", table)


if __name__ == "__main__":
    unittest.main()
