from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from nvs_benchmark.runtime import audit_docstrings, save_doc_audit_report


class DocsAuditTests(unittest.TestCase):
    def test_docs_audit_reports_missing_symbols(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pkg"
            source.mkdir(parents=True, exist_ok=True)
            (source / "mod.py").write_text("def public_fn():\n    return 1\n", encoding="utf-8")

            missing = audit_docstrings(source)
            self.assertGreaterEqual(len(missing), 2)

            report = root / "report.json"
            save_doc_audit_report(missing, report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["missing_count"], len(missing))


if __name__ == "__main__":
    unittest.main()
