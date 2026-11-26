# Reproducible environment

Use **64-bit CPython 3.11.5** for the archived models. `.python-version` records this
requirement; it does not install Python or make a different interpreter compatible.
The development modules permit broader versions, but saved D models deliberately
check the exact Python and numerical-library versions. Do not remove that check.

`requirements.lock` pins all external runtime, test and build dependencies from the
verified environment. `requirements.txt` includes that lock and the three local
editable packages. There are no personal editable paths or API credentials.

From the repository root, verify that `python --version` is 3.11.5, then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pip check
New-Item -ItemType Directory -Force runs | Out-Null
$env:PARTB_ROOT = (Resolve-Path partb).Path
$env:PARTC_ROOT = (Resolve-Path partc).Path
.\.venv\Scripts\python.exe -m pytest parta/tests partb/tests partc/tests partd/tests --import-mode=importlib --basetemp=runs/tests -q -o addopts=
```

The first install supplies the pinned build tools; the second prevents editable
package builds from fetching a different build environment. Use a fresh venv so
unrelated packages do not affect resolution. This lock is validated on Windows
x86-64; other platforms still require their own installation verification.

The build backend is pinned to setuptools 84.2.1, matching the generator recorded
by the original successful editable installation. Older setuptools 65.5.0 wrote
path files that Python 3.11 could not decode in this Chinese Windows checkout;
changing UTF-8 mode alone did not resolve it. Use the pinned build tools above.

Tests use fake credentials. C's autouse fixture removes inherited provider keys
and runtime settings and prevents implicit loading of a repository `.env`.
Tests explicitly checking key precedence set their own fake values. Do not put
real keys in test arguments, test data, XML reports or committed files.

Budget state is compatible with the existing generated version-1 ledgers. A
genuinely absent file starts a new ledger. An existing unreadable, malformed or
inconsistent ledger raises `BudgetStateError` before API initialization in A;
it is never replaced with fresh credit. Preserve the affected file, compare a
backup with the complete call logs and provider usage, then restore only verified
accounting state. Do not delete a ledger or change task IDs to reset spending.
The ledger remains a serial, single-process tool, not a multi-process database.

These production fixes change the SuperAgent source fingerprint. Historical live
results remain evidence for their recorded version; before final submission,
generate report results through complete autonomous runs of the final approved
version. Offline tests and reloaded predictions do not replace those runs.
