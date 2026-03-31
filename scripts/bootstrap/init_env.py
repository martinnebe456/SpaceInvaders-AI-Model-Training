"""Inicializace vyvojoveho prostredi po prvnim naklonovani repozitare."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=["macos", "windows"],
        required=True,
        help="Platforma, pro kterou se ma prostredi pripravit.",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Smaze existujici .venv a vytvori ji znovu.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def ensure_python_version(target_platform: str) -> None:
    version = sys.version_info[:3]
    if version < MIN_PYTHON or version >= MAX_PYTHON:
        fail(
            "Je potreba Python 3.12 nebo 3.13. "
            f"Aktualni interpreter je {version[0]}.{version[1]}.{version[2]}."
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
                "Na Apple Silicon byl detekovan x86_64/Rosetta Python. "
                "Pouzij nativni arm64 Python 3.12 nebo 3.13."
            )

    if target_platform == "windows":
        if platform.architecture()[0] != "64bit":
            fail("Je potreba 64bit Python. 32bit interpreter neni podporovan.")


def recreate_venv_if_requested(recreate: bool) -> None:
    if recreate and VENV_DIR.exists():
        print(f"Mazu existujici virtualni prostredi: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)


def create_venv_if_missing() -> None:
    if VENV_DIR.exists():
        print(f"Pouzivam existujici virtualni prostredi: {VENV_DIR}")
        return

    print(f"Vytvarim virtualni prostredi: {VENV_DIR}")
    subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        cwd=ROOT_DIR,
        check=True,
    )


def venv_python(target_platform: str) -> Path:
    if target_platform == "windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def install_dependencies(target_platform: str) -> None:
    python_bin = venv_python(target_platform)
    if not python_bin.exists():
        fail(f"V .venv nebyl nalezen Python interpreter: {python_bin}")

    run_checked([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
    run_checked(
        [str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    )


def ensure_project_dirs() -> None:
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
    python_bin = venv_python(target_platform)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(ROOT_DIR / ".mplconfig")
    run_checked(
        [str(python_bin), "-m", "py_compile", *sorted(str(p) for p in (ROOT_DIR / "space_invaders_rl").glob("*.py"))],
        env=env,
    )


def print_next_steps(target_platform: str) -> None:
    print()
    print("Hotovo. Prostredi je pripravene.")
    print("Dalsi kroky:")
    if target_platform == "macos":
        print("  1. ./scripts/macos/train_long.sh")
        print("  2. ./scripts/macos/train_cycle.sh")
        print("  3. ./scripts/macos/silent_test.sh")
    else:
        print(r"  1. .\scripts\windows\Train-Long.ps1")
        print(r"  2. .\scripts\windows\Train-Cycle.ps1")
        print(r"  3. .\scripts\windows\Silent-Test.ps1")


def main() -> None:
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
