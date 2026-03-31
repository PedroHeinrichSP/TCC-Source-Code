from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from nvs_benchmark.runtime import RunLogger


class RuntimeLoggingTests(unittest.TestCase):
    def test_run_logger_creates_context_and_events(self) -> None:
        with TemporaryDirectory() as tmp:
            logger = RunLogger(command="unit-test", log_dir=tmp, parameters={"k": "v"})
            logger.event("custom_event", {"ok": True})
            logger.finish("success", {"done": True})

            context = logger.run_dir / "run_context.json"
            events = logger.run_dir / "events.jsonl"
            self.assertTrue(context.exists())
            self.assertTrue(events.exists())

            payload = json.loads(context.read_text(encoding="utf-8"))
            self.assertEqual(payload["command"], "unit-test")
            self.assertEqual(payload["parameters"]["k"], "v")

            lines = events.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
