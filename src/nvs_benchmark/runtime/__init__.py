"""Execução: configuração de ambiente, logs e perfis de hardware."""

from .docs_audit import audit_docstrings, save_doc_audit_report
from .hardware import HardwareSnapshot, detect_hardware_snapshot
from .logging import RunLogger

__all__ = [
	"RunLogger",
	"HardwareSnapshot",
	"detect_hardware_snapshot",
	"audit_docstrings",
	"save_doc_audit_report",
]
