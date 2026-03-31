"""Bootstrap the local development environment after cloning the repository."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 14)


def parse_args() -> argparse.Namespace:
    """Parse the small set of bootstrap options exposed to platform wrappers."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=["macos", "windows"],
        required=True,
        help="Target platform for environment setup.",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete the existing .venv and create it again from scratch.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    """Print a user-facing error and exit with a non-zero status."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def ensure_python_version(target_platform: str) -> None:
    """Reject unsupported Python versions and unsupported platform architectures."""
    version = sys.version_info[:3]
    if version < MIN_PYTHON or version >= MAX_PYTHON:
        fail(
            "Python 3.12 or 3.13 is required. "
            f"The current interpreter is {version[0]}.{version[1]}.{version[2]}."
        )

    if target_platform == "macos":
        host_arch = ""
        try:
            host_arch = subprocess.run(
                ["uname", "-m"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            host_arch = platform.machine()

        python_arch = platform.machine()
        if host_arch == "arm64" and python_arch != "arm64":
            fail(
                "An x86_64/Rosetta Python was detected on Apple Silicon. "
                "Use a native arm64 Python 3.12 or 3.13 build instead."
            )

    if target_platform == "windows":
        if platform.architecture()[0] != "64bit":
            fail("A 64-bit Python interpreter is required. 32-bit Python is not supported.")


def recreate_venv_if_requested(recreate: bool) -> None:
    """Delete the virtual environment only when the caller explicitly asks for it."""
    if recreate and VENV_DIR.exists():
        print(f"Removing existing virtual environment: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)


def create_venv_if_missing() -> None:
    """Create .venv once and reuse it on later bootstrap runs."""
    if VENV_DIR.exists():
        print(f"Reusing existing virtual environment: {VENV_DIR}")
        return

    print(f"Creating virtual environment: {VENV_DIR}")
    subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        cwd=ROOT_DIR,
        check=True,
    )


def venv_python(target_platform: str) -> Path:
    """Return the Python executable path inside .venv for the given platform."""
    if target_platform == "windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command in the repo root and fail fast if it exits with an error."""
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def install_dependencies(target_platform: str) -> None:
    """Install the Python dependencies inside the prepared virtual environment."""
    python_bin = venv_python(target_platform)
    if not python_bin.exists():
        fail(f"Python was not found inside .venv: {python_bin}")

    run_checked([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
    run_checked(
        [str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    )


def ensure_project_dirs() -> None:
    """Create local-only runtime directories expected by the training scripts."""
    for relative in [
        ".mplconfig",
        "runs/automation",
        "runs/space_invaders_dqn",
        "models/checkpoints",
        "models/backups",
    ]:
        path = ROOT_DIR / relative
        path.mkdir(parents=True, exist_ok=True)


def run_sanity_checks(target_platform: str) -> None:
    """Compile the Python package once to catch syntax errors early."""
    python_bin = venv_python(target_platform)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(ROOT_DIR / ".mplconfig")
    run_checked(
        [
            str(python_bin),
            "-m",
            "py_compile",
            *sorted(str(p) for p in (ROOT_DIR / "space_invaders_rl").glob("*.py")),
        ],
        env=env,
    )


def print_next_steps(target_platform: str) -> None:
    """Print the most common follow-up commands after a successful bootstrap."""
    print()
    print("Done. The environment is ready.")
    print("Next steps:")
    if target_platform == "macos":
        print("  1. ./scripts/macos/train_long.sh")
        print("  2. ./scripts/macos/train_cycle.sh")
        print("  3. ./scripts/macos/silent_test.sh")
    else:
        print(r"  1. .\scripts\windows\Train-Long.ps1")
        print(r"  2. .\scripts\windows\Train-Cycle.ps1")
        print(r"  3. .\scripts\windows\Silent-Test.ps1")


def main() -> None:
    """Run the full bootstrap sequence from validation to sanity checks."""
    args = parse_args()
    os.chdir(ROOT_DIR)
    ensure_python_version(args.platform)
    recreate_venv_if_requested(args.recreate_venv)
    create_venv_if_missing()
    install_dependencies(args.platform)
    ensure_project_dirs()
    run_sanity_checks(args.platform)
    print_next_steps(args.platform)


if __name__ == "__main__":
    main()
