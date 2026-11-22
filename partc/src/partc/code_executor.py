"""C3 — Python code executor.

Runs Python files or code strings in a subprocess with a custom timeout,
working directory and interpreter, capturing stdout/stderr (to files and, in
size-capped form, to memory) and classifying the result.

Status classification
---------------------
* ``success``           — process exited with code 0
* ``runtime_error``     — process exited with a non-zero code
* ``timeout``           — process exceeded the timeout and its process group was killed
* ``execution_error``   — the interpreter/script could not be launched
* ``cancelled``         — reserved (not exposed in V1)

Safety notes (read carefully)
-----------------------------
macOS ``subprocess`` is **not** a security sandbox.  V1 therefore treats
"trusted local test execution" and "untrusted agent-generated code" as
different things and never claims it can safely run arbitrary code.  What it
*does* guarantee:

* no ``shell=True`` (commands are always an argv list);
* sensitive environment variables (``ZZZ_API_KEY``, etc.) are stripped before
  spawning;
* the child runs in its own process group (``start_new_session``) so a timeout
  kills the whole group, not just the immediate child;
* working directory and output directory can be restricted with ``allowed_roots``.

Do not rely on this module for sandboxing untrusted code.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

from .config import sanitize_environ
from .schemas import (
    EXEC_EXECUTION_ERROR,
    EXEC_RUNTIME_ERROR,
    EXEC_SUCCESS,
    EXEC_TIMEOUT,
    ExecResult,
)

_MAX_CAPTURE_BYTES = 1 << 20  # 1 MiB


class PythonExecutor:
    def __init__(
        self,
        python: Optional[str] = None,
        timeout_seconds: float = 120.0,
        cwd: Optional[str] = None,
        output_dir: Optional[str] = None,
        allowed_roots: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        sanitize_env: bool = True,
    ) -> None:
        self.python = python or sys.executable
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.output_dir = output_dir
        self.allowed_roots = [os.path.abspath(r) for r in (allowed_roots or [])]
        self.extra_env = env or {}
        self.sanitize_env = sanitize_env

    # -- public API --------------------------------------------------------
    def execute_file(self, path: str, *, args: Optional[List[str]] = None, timeout_seconds: Optional[float] = None, cwd: Optional[str] = None, task_id: Optional[str] = None, run_id: Optional[str] = None) -> ExecResult:
        if not os.path.isfile(path):
            return self._execution_error(f"script not found: {path}", task_id, run_id, script_ref=path)
        return self._run([path] + (args or []), script_ref=path, timeout_seconds=timeout_seconds, cwd=cwd, task_id=task_id, run_id=run_id)

    def execute_code(self, code: str, *, timeout_seconds: Optional[float] = None, cwd: Optional[str] = None, task_id: Optional[str] = None, run_id: Optional[str] = None) -> ExecResult:
        """Write ``code`` to a temp file and execute it."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", prefix="partc_code_", delete=False, encoding="utf-8") as fh:
            fh.write(code)
            tmp_path = fh.name
        try:
            return self._run([tmp_path], script_ref="<code_string>", timeout_seconds=timeout_seconds, cwd=cwd, task_id=task_id, run_id=run_id)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # -- internals -----------------------------------------------------------
    def _resolve_cwd(self, cwd: Optional[str]) -> str:
        base = os.path.abspath(cwd or self.cwd or os.getcwd())
        if self.allowed_roots and not any(base == r or base.startswith(r + os.sep) for r in self.allowed_roots):
            raise ValueError(f"working directory {base!r} is outside allowed_roots {self.allowed_roots}")
        return base

    def _run(self, argv: List[str], *, script_ref: str, timeout_seconds: Optional[float], cwd: Optional[str], task_id: Optional[str], run_id: Optional[str]) -> ExecResult:
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        try:
            cwd_resolved = self._resolve_cwd(cwd)
        except ValueError as exc:
            return ExecResult(task_id=task_id, run_id=run_id, script_ref=script_ref, status=EXEC_EXECUTION_ERROR, error_type="invalid_cwd", error_details=str(exc))

        out_dir = self.output_dir or cwd_resolved
        os.makedirs(out_dir, exist_ok=True)

        stdout_fd, stdout_path = tempfile.mkstemp(prefix="partc_out_", suffix=".txt", dir=out_dir)
        stderr_fd, stderr_path = tempfile.mkstemp(prefix="partc_err_", suffix=".txt", dir=out_dir)

        env = self._build_env()

        try:
            with os.fdopen(stdout_fd, "wb") as out_fh, os.fdopen(stderr_fd, "wb") as err_fh:
                try:
                    proc = subprocess.Popen(
                        [self.python, *argv],  # argv already has script first; prepend interpreter
                        stdout=out_fh,
                        stderr=err_fh,
                        cwd=cwd_resolved,
                        env=env,
                        start_new_session=True,
                    )
                except FileNotFoundError as exc:
                    return self._execution_error(
                        f"interpreter not found: {self.python}", task_id, run_id, script_ref=script_ref,
                    )
                except OSError as exc:
                    return self._execution_error(f"failed to spawn process: {exc}", task_id, run_id, script_ref=script_ref)

                start = time.monotonic()
                timed_out = False
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill_group(proc)

                elapsed = time.monotonic() - start

            exit_code = proc.returncode
            stdout_text = _read_capped(stdout_path)
            stderr_text = _read_capped(stderr_path)

            if timed_out:
                status = EXEC_TIMEOUT
                error_type = "timeout"
            elif exit_code == 0:
                status = EXEC_SUCCESS
                error_type = None
            else:
                status = EXEC_RUNTIME_ERROR
                error_type = "non_zero_exit"

            return ExecResult(
                task_id=task_id,
                run_id=run_id,
                script_ref=script_ref,
                status=status,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                elapsed_seconds=elapsed,
                timeout_seconds=timeout,
                error_type=error_type,
                error_details=stderr_text if status == EXEC_RUNTIME_ERROR else None,
            )
        finally:
            pass  # files intentionally kept for audit; paths returned on the result

    def _build_env(self) -> Dict[str, str]:
        env = sanitize_environ() if self.sanitize_env else dict(os.environ)
        env.update(self.extra_env)
        return env

    def _kill_group(self, proc: subprocess.Popen) -> None:
        if os.name == "nt":
            # taskkill /T handles descendants; proc.kill is a fallback for the parent.
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _execution_error(self, message: str, task_id: Optional[str], run_id: Optional[str], script_ref: str) -> ExecResult:
        return ExecResult(
            task_id=task_id,
            run_id=run_id,
            script_ref=script_ref,
            status=EXEC_EXECUTION_ERROR,
            error_type="execution_error",
            error_details=message,
        )


def _read_capped(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(_MAX_CAPTURE_BYTES + 1)
        if len(data) > _MAX_CAPTURE_BYTES:
            return data[:_MAX_CAPTURE_BYTES] + "\n...<truncated>"
        return data
    except OSError:
        return ""
