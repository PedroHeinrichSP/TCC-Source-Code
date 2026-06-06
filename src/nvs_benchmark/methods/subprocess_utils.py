"""Utilitários para execução robusta de subprocessos externos.

Fornece:
- Execução com timeout configurável
- Detecção de Out-Of-Memory (OOM) em saída de GPU
- Captura de tracebacks completos de subprocessos
- Mensagens de erro amigáveis para estudantes
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Padrões de texto que indicam OOM de GPU em diferentes frameworks
# ---------------------------------------------------------------------------
_OOM_SIGNATURES: list[str] = [
    "out of memory",
    "CUDA out of memory",
    "OOM",
    "RuntimeError: CUDA",
    "torch.cuda.OutOfMemoryError",
    "cublas",
    "cusolver",
    "device-side assert triggered",
    "cudaErrorMemoryAllocation",
    "insufficient memory",
    "not enough memory",
    "CUDAOutOfMemoryError",
]


# ---------------------------------------------------------------------------
# Resultado de um subprocesso
# ---------------------------------------------------------------------------
@dataclass
class SubprocessResult:
    """Resultado completo de execução de subprocesso externo."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    oom_detected: bool = False
    oom_hint: str = ""
    error_traceback: str = ""

    @property
    def success(self) -> bool:
        """Verdadeiro quando o processo encerrou com código 0."""
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Detecção de OOM
# ---------------------------------------------------------------------------
def _detect_oom(stdout: str, stderr: str) -> tuple[bool, str]:
    """Detecta padrões de Out-Of-Memory em saída do processo.

    Retorna (encontrou_oom, mensagem_dica).
    """
    combined = (stdout + "\n" + stderr).lower()
    for sig in _OOM_SIGNATURES:
        if sig.lower() in combined:
            hint = (
                "ERRO DE MEMORIA NA GPU (OOM): O processo esgotou a VRAM disponivel.\n"
                "Sugestoes para resolver:\n"
                "  1. Reduza a resolucao das imagens (ex: --downscale_factor 2 ou 4)\n"
                "  2. Diminua o batch_size nas configuracoes do metodo\n"
                "  3. Use o preset 'smoke' ou 'quick' com menos iteracoes\n"
                "  4. Feche outros programas que estejam usando a GPU\n"
                f"  (Detectado: '{sig}')"
            )
            return True, hint
    return False, ""


# ---------------------------------------------------------------------------
# Extração de traceback
# ---------------------------------------------------------------------------
def _extract_traceback(stderr: str) -> str:
    """Extrai o traceback Python mais recente da saída de erro.

    Retorna a string do traceback ou string vazia se não encontrado.
    """
    lines = stderr.splitlines()
    tb_lines: list[str] = []
    in_tb = False
    for line in lines:
        if line.strip().startswith("Traceback (most recent call last):"):
            in_tb = True
            tb_lines = [line]
        elif in_tb:
            tb_lines.append(line)
            # Traceback termina quando encontramos uma linha que não é indentada
            # e não é "File ..." ou "  ..."
            if tb_lines and not line.startswith(" ") and not line.startswith("\t") and line.strip():
                if line.startswith("Traceback"):
                    # Novo traceback, reiniciar
                    tb_lines = [line]
                elif "Error" in line or "Exception" in line:
                    # Última linha do traceback
                    break

    return "\n".join(tb_lines) if tb_lines else ""


# ---------------------------------------------------------------------------
# Executor principal
# ---------------------------------------------------------------------------
def run_subprocess(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> SubprocessResult:
    """Executa um subprocesso externo com robustez e rastreabilidade.

    Args:
        command: Sequência de strings com o comando a executar.
        cwd: Diretório de trabalho do processo.
        timeout: Tempo máximo em segundos. None = sem limite.
        env: Variáveis de ambiente adicionais (mescladas com o ambiente atual).
        capture_output: Se True, captura stdout/stderr; senão, usa herdados.

    Returns:
        SubprocessResult com código de retorno, saída, OOM e traceback.
    """
    import os

    effective_env: dict[str, str] | None = None
    if env:
        effective_env = {**os.environ, **env}

    cmd_list = [str(c) for c in command]
    start = time.perf_counter()
    timed_out = False
    stdout_text = ""
    stderr_text = ""
    returncode = -1

    try:
        proc = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            env=effective_env,
        )
        returncode = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""

    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout_text = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_text = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr_text += f"\n[TIMEOUT] Processo encerrado apos {timeout}s."

    except FileNotFoundError as exc:
        returncode = 127
        stderr_text = (
            f"[ERRO] Comando nao encontrado: {cmd_list[0]}\n"
            f"Verifique se o executavel esta no PATH.\nDetalhe: {exc}"
        )

    except Exception as exc:  # noqa: BLE001
        returncode = -1
        stderr_text = f"[ERRO] Falha inesperada ao iniciar subprocesso: {exc}"

    duration = time.perf_counter() - start
    oom_detected, oom_hint = _detect_oom(stdout_text, stderr_text)
    traceback_text = _extract_traceback(stderr_text)

    return SubprocessResult(
        command=cmd_list,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        duration_seconds=duration,
        timed_out=timed_out,
        oom_detected=oom_detected,
        oom_hint=oom_hint,
        error_traceback=traceback_text,
    )


def run_subprocess_streaming(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Executa subprocesso transmitindo a saida em tempo real para stdout."""
    import os

    effective_env: dict[str, str] = dict(os.environ)
    effective_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        effective_env.update(env)

    cmd_list = [str(c) for c in command]
    start = time.perf_counter()
    timed_out = False
    returncode = -1
    combined_chunks: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=effective_env,
            bufsize=1,
        )
        stream = proc.stdout
        if stream is None:
            raise RuntimeError("Processo iniciou sem stdout capturavel.")

        current_chunk: list[str] = []
        while True:
            if timeout is not None and (time.perf_counter() - start) > timeout:
                timed_out = True
                proc.kill()
                break

            char = stream.read(1)
            if char == "":
                if proc.poll() is not None:
                    break
                continue

            if char in ("\r", "\n"):
                if current_chunk:
                    line = "".join(current_chunk)
                    print(line, flush=True)
                    combined_chunks.append(line + "\n")
                    current_chunk = []
            else:
                current_chunk.append(char)

        if current_chunk:
            line = "".join(current_chunk)
            print(line, flush=True)
            combined_chunks.append(line + "\n")

        remaining_stdout, _ = proc.communicate()
        if remaining_stdout:
            print(remaining_stdout, end="" if remaining_stdout.endswith("\n") else "\n", flush=True)
            combined_chunks.append(remaining_stdout)

        returncode = proc.returncode if proc.returncode is not None else -1
        if timed_out:
            returncode = -1
            combined_chunks.append(f"\n[TIMEOUT] Processo encerrado apos {timeout}s.\n")

    except FileNotFoundError as exc:
        returncode = 127
        combined_chunks.append(
            f"[ERRO] Comando nao encontrado: {cmd_list[0]}\n"
            f"Verifique se o executavel esta no PATH.\nDetalhe: {exc}\n"
        )

    except Exception as exc:  # noqa: BLE001
        returncode = -1
        combined_chunks.append(f"[ERRO] Falha inesperada ao iniciar subprocesso: {exc}\n")

    combined_text = "".join(combined_chunks)
    duration = time.perf_counter() - start
    oom_detected, oom_hint = _detect_oom(combined_text, combined_text)
    traceback_text = _extract_traceback(combined_text)

    return SubprocessResult(
        command=cmd_list,
        returncode=returncode,
        stdout=combined_text,
        stderr=combined_text,
        duration_seconds=duration,
        timed_out=timed_out,
        oom_detected=oom_detected,
        oom_hint=oom_hint,
        error_traceback=traceback_text,
    )


def format_subprocess_error(result: SubprocessResult) -> str:
    """Formata mensagem de erro legível para exibir ao estudante.

    Prioriza: OOM → timeout → traceback → stderr genérico.
    """
    if result.oom_detected:
        return result.oom_hint

    if result.timed_out:
        elapsed = f"{result.duration_seconds:.1f}s"
        return (
            f"TIMEOUT: O processo demorou mais do que o limite configurado ({elapsed}).\n"
            "Sugestoes:\n"
            "  1. Use um preset com menos iteracoes (ex: --preset smoke)\n"
            "  2. Aumente o timeout com --timeout em configs\n"
            "  3. Verifique se a GPU esta disponivel (pode ter caido em modo CPU)"
        )

    if result.error_traceback:
        return f"Traceback do subprocesso:\n{result.error_traceback}"

    if result.stderr.strip():
        # Retornar as últimas 20 linhas do stderr
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
        return f"Saida de erro (ultimas linhas):\n{stderr_tail}"

    return f"Processo falhou com codigo {result.returncode} (sem mensagem de erro capturada)."
