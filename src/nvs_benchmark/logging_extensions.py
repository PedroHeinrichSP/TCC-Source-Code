"""Extensões para o sistema de logging estruturado.

Fornece:
- Progress bar e indicadores de progresso
- Resumo final de execução
- Callbacks para eventos importantes
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class ExecutionSummary:
    """Resumo de uma execução completa."""
    run_id: str
    method: str
    dataset: str
    command: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    status: str = "running"  # running, success, failed
    error_message: Optional[str] = None
    train_seconds: float = 0.0
    inference_seconds: float = 0.0
    metrics: Optional[Dict[str, Any]] = None
    output_files: Optional[Dict[str, str]] = None  # nome -> caminho
    

class ExecutionTracker:
    """Rastreia progresso de uma execução."""
    
    def __init__(self, run_id: str, method: str, dataset: str, command: str):
        """Inicializa rastreador de execução."""
        self.run_id = run_id
        self.method = method
        self.dataset = dataset
        self.command = command
        self.start_time = datetime.now()
        self.summary = ExecutionSummary(
            run_id=run_id,
            method=method,
            dataset=dataset,
            command=command,
            start_time=self.start_time.isoformat(),
        )
        self.steps: list[Dict[str,  Any]] = []
        self.output_files: Dict[str, str] = {}
        
    def log_step(self, step_name: str, status: str = "completed", details: Optional[Dict] = None) -> None:
        """Registra conclusão de um passo (treino, inferência, etc)."""
        step_info = {
            "name": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            step_info.update(details)
        self.steps.append(step_info)
    
    def add_output_file(self, description: str, filepath: str) -> None:
        """Registra um arquivo de saída importante."""
        self.output_files[description] = filepath
        self.summary.output_files = self.output_files
    
    def set_training_metrics(self, duration_seconds: float) -> None:
        """Define duração do treinamento."""
        self.summary.train_seconds = duration_seconds
        self.log_step("training", "completed", {"duration_seconds": duration_seconds})
    
    def set_inference_metrics(self, duration_seconds: float) -> None:
        """Define duração da inferência."""
        self.summary.inference_seconds = duration_seconds
        self.log_step("inference", "completed", {"duration_seconds": duration_seconds})
    
    def set_quality_metrics(self, metrics: Dict[str, Any]) -> None:
        """Define métricas de qualidade (PSNR, SSIM, etc)."""
        self.summary.metrics = metrics
        self.log_step("metrics", "completed", metrics)
    
    def finish_success(self) -> None:
        """Marca execução como bem-sucedida."""
        self.summary.status = "success"
        self.summary.end_time = datetime.now().isoformat()
        delta = datetime.now() - self.start_time
        self.summary.duration_seconds = delta.total_seconds()
    
    def finish_failed(self, error_message: str) -> None:
        """Marca execução como falhada."""
        self.summary.status = "failed"
        self.summary.error_message = error_message
        self.summary.end_time = datetime.now().isoformat()
        delta = datetime.now() - self.start_time
        self.summary.duration_seconds = delta.total_seconds()
    
    def save_summary(self, output_path: str) -> None:
        """Salva resumo em arquivo JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = asdict(self.summary)
        summary_data["steps"] = self.steps
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self) -> None:
        """Imprime resumo da execução no console."""
        from nvs_benchmark.cli_extensions import print_separator
        
        print_separator("=")
        print("📋 Resumo da Execução")
        print_separator("=")
        print()
        
        print(f"Run ID: {self.run_id}")
        print(f"Método: {self.method}")
        print(f"Dataset: {self.dataset}")
        print(f"Comando: {self.command}")
        print(f"Status: {self._status_emoji(self.summary.status)} {self.summary.status.upper()}")
        print()
        
        print("⏱️  Tempos:")
        print(f"  Duração total: {self._format_duration(self.summary.duration_seconds)}")
        if self.summary.train_seconds > 0:
            print(f"  Treino: {self._format_duration(self.summary.train_seconds)}")
        if self.summary.inference_seconds > 0:
            print(f"  Inferência: {self._format_duration(self.summary.inference_seconds)}")
        print()
        
        if self.summary.metrics:
            print("📊 Métricas:")
            for key, value in self.summary.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        if self.output_files:
            print("📁 Outputs:")
            for desc, path in self.output_files.items():
                print(f"  {desc}: {path}")
            print()
        
        if self.summary.error_message:
            print(f"❌ Erro: {self.summary.error_message}")
            print()
        
        print_separator("=")
    
    @staticmethod
    def _status_emoji(status: str) -> str:
        """Retorna emoji para status."""
        emojis = {
            "success": "✓",
            "failed": "✗",
            "running": "⏳",
        }
        return emojis.get(status, "?")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formata duração em formato legível."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class ProgressCallback:
    """Callback para processos com progr esso."""
    
    def __init__(self, total_steps: int, description: str = "Progresso"):
        """Inicializa callback de progresso."""
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
    
    def update(self, step: int, message: str = "") -> None:
        """Atualiza progresso."""
        self.current_step = step
        percentage = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
        bar_length = 40
        filled = int(bar_length * step / self.total_steps) if self.total_steps > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r[{bar}] {percentage:.1f}% ({step}/{self.total_steps}) {message}", end="", flush=True)
    
    def finish(self) -> None:
        """Finaliza a barra de progresso."""
        print()  # Nova linha


class MetricsCollector:
    """Coleta métricas durante execução."""
    
    def __init__(self):
        """Inicializa coletor de métricas."""
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
    
    def record_metric(self, name: str, value: Any, timestamp: Optional[datetime] = None) -> None:
        """Registra uma métrica."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": timestamp.isoformat(),
        })
    
    def get_latest(self, name: str) -> Any:
        """Obtém último valor registrado para uma métrica."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]["value"]
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                numeric_values = [v["value"] for v in values if isinstance(v["value"], (int, float))]
                if numeric_values:
                    summary[name] = {
                        "latest": numeric_values[-1],
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "average": sum(numeric_values) / len(numeric_values),
                    }
        return summary
    
    def save_metrics(self, output_path: str) -> None:
        """Salva métricas em JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
