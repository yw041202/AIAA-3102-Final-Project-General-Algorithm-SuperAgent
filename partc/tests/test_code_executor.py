"""Code executor tests: success, runtime error, timeout, child cleanup,
execution error, and env sanitisation."""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from partc.code_executor import PythonExecutor
from partc.schemas import (
    EXEC_EXECUTION_ERROR,
    EXEC_RUNTIME_ERROR,
    EXEC_SUCCESS,
    EXEC_TIMEOUT,
)


@pytest.fixture
def executor(tmp_path):
    return PythonExecutor(
        python=sys.executable,
        timeout_seconds=10.0,
        cwd=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        sanitize_env=True,
    )


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied, not a missing PID.
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_run_success(executor):
    r = executor.execute_code("print('hello world')", task_id="t1")
    assert r.status == EXEC_SUCCESS
    assert r.exit_code == 0
    assert "hello world" in r.stdout
    assert os.path.isfile(r.stdout_path)


def test_runtime_error(executor):
    r = executor.execute_code("raise ValueError('boom')")
    assert r.status == EXEC_RUNTIME_ERROR
    assert r.exit_code != 0
    assert "boom" in r.stderr


def test_timeout(executor):
    r = executor.execute_code("import time; time.sleep(30)", timeout_seconds=0.5)
    assert r.status == EXEC_TIMEOUT
    assert r.error_type == "timeout"


def test_child_process_group_cleaned_up(executor, tmp_path):
    pidfile = str(tmp_path / "pids.txt")
    code = (
        "import os, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({pidfile!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(60)\n"
    )
    r = executor.execute_code(code, timeout_seconds=1.0)
    assert r.status == EXEC_TIMEOUT

    for _ in range(20):
        if os.path.isfile(pidfile):
            break
        time.sleep(0.1)

    assert os.path.isfile(pidfile), "script never wrote its pid file"
    parent_pid, child_pid = (int(x) for x in open(pidfile).read().split())
    # both parent and its child must be gone once the group was signalled
    deadline = time.time() + 5
    while time.time() < deadline and (_pid_exists(parent_pid) or _pid_exists(child_pid)):
        time.sleep(0.1)
    assert not _pid_exists(parent_pid)
    assert not _pid_exists(child_pid)


def test_execution_error_missing_script(executor):
    r = executor.execute_file("/no/such/script.py")
    assert r.status == EXEC_EXECUTION_ERROR


def test_execution_error_bad_interpreter(tmp_path):
    ex = PythonExecutor(python="/no/such/python", timeout_seconds=5, cwd=str(tmp_path))
    r = ex.execute_code("print(1)")
    assert r.status == EXEC_EXECUTION_ERROR


def test_env_does_not_leak_sensitive_vars(executor, monkeypatch):
    monkeypatch.setenv("ZZZ_API_KEY", "sk-secret-123")
    code = (
        "import os\n"
        "print('ZZZ_API_KEY' in os.environ)\n"
        "print(os.environ.get('ZZZ_API_KEY'))\n"
    )
    r = executor.execute_code(code)
    assert r.status == EXEC_SUCCESS
    assert "False" in r.stdout
    assert "sk-secret-123" not in r.stdout


def test_no_shell_interpretation(executor):
    # Shell metacharacters must reach Python as literal source (=> syntax error),
    # never be evaluated by a shell (=> `echo` output / exit 0).
    r = executor.execute_code("echo $(id) && echo hello")
    assert r.status == EXEC_RUNTIME_ERROR
    assert r.exit_code != 0
