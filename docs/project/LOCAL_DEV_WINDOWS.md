# Local dev on Windows

This runbook covers setting up a working Python environment on a Windows laptop
so you can run the test suite, iterate on `src/`, and validate a PR before it
goes to the Technion VM. Follow it exactly if your workspace lives inside
OneDrive; the workarounds here were learned the hard way.

Heavy CUT3R extraction still belongs on the Technion VM. See
`docs/project/TECHNION_VM_RUNBOOK.md`. This file is for the pre-VM loop:
editing code, running unit tests, checking imports.

## Prerequisites

- Windows 10 or 11 with the `py` launcher installed
  (comes with any python.org installer).
- **Python 3.11 or 3.12** available to `py`. Verify with `py -0`:

  ```powershell
  py -0
  ```

  If the list does not include a `-3.11-64` or `-3.12-64` entry, install one
  from <https://www.python.org/downloads/>. Do not rely on the Windows Store
  `python3.exe` stub — it is not a real interpreter.

- **Avoid Python 3.13 for now.** `numpy==1.26.4` (pinned in `pyproject.toml`)
  has no official cp313 wheel; the wheel PyPI serves for it on Windows is
  MinGW-w64 experimental and crashes inside subprocess calls
  (`test_cut3r_provenance::test_compatibility_patch_is_applied_and_validated`
  fails with `0xC0000005`). Track: pin `python = ">=3.11,<3.13"` in a future PR
  or bump numpy to 2.x.

## Create the venv outside OneDrive

If the repo lives inside OneDrive (default for Technion accounts) the venv
**must** live somewhere else. Two independent Windows problems otherwise bite:

1. `pip` cannot reliably replace files inside an OneDrive-tracked directory
   (`[Errno 22] Invalid argument` during `setuptools 58 → 83` uninstall).
2. `torch`'s bundled license paths exceed the 260-character `MAX_PATH`
   limit when the base workspace path is long (`WinError 206`).

Use a short path such as `C:\dev\venvs\cut3r\`:

```powershell
if (-not (Test-Path C:\dev\venvs)) { New-Item -ItemType Directory C:\dev\venvs | Out-Null }
py -3.12 -m venv --upgrade-deps C:\dev\venvs\cut3r
C:\dev\venvs\cut3r\Scripts\python.exe --version
```

`--upgrade-deps` is important: it pulls a current `pip` and `setuptools` into
the venv at creation time so pip never has to uninstall the old bundled
setuptools later.

## Install the project

From the repo root:

```powershell
C:\dev\venvs\cut3r\Scripts\python.exe -m pip install -e ".[dev]"
```

Then point VS Code at the interpreter: Command Palette →
**Python: Select Interpreter** → **Enter interpreter path** →
`C:\dev\venvs\cut3r\Scripts\python.exe`.

Nothing in the repo (`.gitignore`, `pyproject.toml`, VS Code workspace
settings) references this path; it is per-machine.

## Run the tests

```powershell
C:\dev\venvs\cut3r\Scripts\python.exe -m pytest tests/ -q
```

Expected on Python 3.12: **55/55 pass**.
Expected on Python 3.13 today: **54/55 pass**, with the numpy-related failure
called out above.

If `pytest` itself fails to collect (import errors), the flatten introduced in
branch `codebaseNdatapipeline-redesign` requires that `src/`, `scripts/`, and
`tests/` are at the repository root, not under `data_pipeline/`. Check
`pyproject.toml` — `[tool.setuptools.packages.find].where` should be `["."]`.

## When OneDrive is still involved

Because your workspace itself is inside OneDrive, occasional sync interference
can still lock a file mid-write during `pip install` or a rebase. If a
command errors with `Errno 22` or `Access is denied`:

1. Right-click the OneDrive tray icon → **Pause syncing** → **2 hours**.
2. Retry the command.
3. Resume sync when finished.

The two long-term fixes are outside this runbook's scope:

- Enable Windows long-path support (registry key
  `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`;
  needs admin and a re-login).
- Move the workspace itself outside OneDrive (any short path such as
  `C:\dev\cut3r-semantic-probing\`). Recommended if you do this often.
