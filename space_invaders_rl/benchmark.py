"""Benchmark ruznych treningovych konfiguraci pro Space Invaders RL."""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


FPS_PATTERN = re.compile(r"\|\s+fps\s+\|\s+([0-9.]+)\s+\|")


@dataclass
class RunResult:
    n_envs: int
    repeat: int
    elapsed_seconds: float
    steps_per_second: float
    last_logged_fps: float | None
    load_before: tuple[float, float, float]
    load_after: tuple[float, float, float]
    log_path: Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=[1, 4, 6, 8, 10],
        help="Seznam hodnot n_envs, ktere se maji otestovat.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=80_000,
        help="Kolik kroku ma trvat jeden mereny beh.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Kolikrat zopakovat kazdou konfiguraci.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Limit kroku jedne epizody behem benchmarku.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=12,
        help="Pauza mezi behy, aby si stroj trochu oddechl.",
    )
    parser.add_argument(
        "--max-load-ratio",
        type=float,
        default=0.75,
        help="Pokud je 1m loadavg / pocet CPU nad timto prahem, benchmark pocka.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=20,
        help="Jak dlouho cekat mezi kontrolami zatizeni.",
    )
    parser.add_argument(
        "--max-waits",
        type=int,
        default=6,
        help="Kolikrat maximalne cekat na uvolneni systemu.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("runs/benchmark"),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Ponecha docasne modely a replay buffery z benchmarku.",
    )
    return parser.parse_args()


def cpu_count() -> int:
    return os.cpu_count() or 1


def get_loadavg() -> tuple[float, float, float]:
    if not hasattr(os, "getloadavg"):
        return (0.0, 0.0, 0.0)

    try:
        return os.getloadavg()
    except OSError:
        return (0.0, 0.0, 0.0)


def load_ratio(load: tuple[float, float, float]) -> float:
    return load[0] / cpu_count()


def wait_for_reasonable_load(args) -> tuple[float, float, float]:
    current_load = get_loadavg()
    if current_load == (0.0, 0.0, 0.0):
        return current_load

    attempts = 0
    while load_ratio(current_load) > args.max_load_ratio and attempts < args.max_waits:
        ratio = load_ratio(current_load)
        print(
            "System je aktualne dost vytizeny: "
            f"load1={current_load[0]:.2f}, ratio={ratio:.2f}. "
            f"Cekam {args.wait_seconds}s..."
        )
        time.sleep(args.wait_seconds)
        current_load = get_loadavg()
        attempts += 1

    return current_load


def benchmark_command(
    args,
    run_dir: Path,
    n_envs: int,
    repeat: int,
) -> list[str]:
    vec_env = args.vec_env
    if vec_env == "auto":
        vec_env = "dummy" if n_envs == 1 else "subproc"

    model_path = run_dir / f"model_env{n_envs}_run{repeat}"
    replay_path = run_dir / f"model_env{n_envs}_run{repeat}_replay.pkl"
    checkpoint_dir = run_dir / f"checkpoints_env{n_envs}_run{repeat}"
    tensorboard_log = run_dir / "tensorboard"

    return [
        sys.executable,
        "-m",
        "space_invaders_rl.train",
        "--timesteps",
        str(args.timesteps),
        "--max-steps",
        str(args.max_steps),
        "--n-envs",
        str(n_envs),
        "--vec-env",
        vec_env,
        "--device",
        args.device,
        "--seed",
        str(args.seed + repeat),
        "--batch-size",
        str(args.batch_size),
        "--model-path",
        str(model_path),
        "--replay-buffer-path",
        str(replay_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--tensorboard-log",
        str(tensorboard_log),
        "--skip-check-env",
        "--fresh-start",
    ]


def run_single_benchmark(args, run_dir: Path, n_envs: int, repeat: int) -> RunResult:
    run_name = f"env{n_envs}_run{repeat}"
    log_path = run_dir / f"{run_name}.log"
    before_load = wait_for_reasonable_load(args)
    command = benchmark_command(args, run_dir, n_envs, repeat)

    print(
        f"Spoustim benchmark {run_name}: "
        f"n_envs={n_envs}, timesteps={args.timesteps}, "
        f"vec_env={command[command.index('--vec-env') + 1]}, device={args.device}"
    )
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            check=True,
        )
        output = completed.stdout + completed.stderr
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        raise RuntimeError(
            f"Benchmark {run_name} selhal. Viz log {log_path}"
        ) from exc

    elapsed = time.perf_counter() - start
    after_load = get_loadavg()
    log_path.write_text(output, encoding="utf-8")

    logged_fps_matches = FPS_PATTERN.findall(output)
    last_logged_fps = float(logged_fps_matches[-1]) if logged_fps_matches else None
    steps_per_second = args.timesteps / elapsed

    print(
        f"Hotovo {run_name}: elapsed={elapsed:.1f}s, "
        f"measured_steps_per_sec={steps_per_second:.1f}, "
        f"last_logged_fps={last_logged_fps if last_logged_fps is not None else 'n/a'}"
    )

    if not args.keep_artifacts:
        cleanup_run_artifacts(run_dir, n_envs, repeat)

    return RunResult(
        n_envs=n_envs,
        repeat=repeat,
        elapsed_seconds=elapsed,
        steps_per_second=steps_per_second,
        last_logged_fps=last_logged_fps,
        load_before=before_load,
        load_after=after_load,
        log_path=log_path,
    )


def cleanup_run_artifacts(run_dir: Path, n_envs: int, repeat: int) -> None:
    model_base = run_dir / f"model_env{n_envs}_run{repeat}"
    replay_path = run_dir / f"model_env{n_envs}_run{repeat}_replay.pkl"
    checkpoint_dir = run_dir / f"checkpoints_env{n_envs}_run{repeat}"

    for path in [model_base.with_suffix(".zip"), replay_path]:
        if path.exists():
            path.unlink()

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def summarize(results: list[RunResult]) -> None:
    grouped: dict[int, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.n_envs, []).append(result)

    print()
    print("=== Benchmark Summary ===")
    print(
        "n_envs | median_steps_per_sec | mean_steps_per_sec | "
        "median_logged_fps | avg_load_before | avg_load_after"
    )

    ranking: list[tuple[float, int]] = []
    for n_envs in sorted(grouped):
        runs = grouped[n_envs]
        measured = [run.steps_per_second for run in runs]
        logged = [run.last_logged_fps for run in runs if run.last_logged_fps is not None]
        avg_load_before = statistics.mean(load_ratio(run.load_before) for run in runs)
        avg_load_after = statistics.mean(load_ratio(run.load_after) for run in runs)
        median_steps = statistics.median(measured)
        mean_steps = statistics.mean(measured)
        median_logged = statistics.median(logged) if logged else float("nan")
        ranking.append((median_steps, n_envs))
        print(
            f"{n_envs:>6} | {median_steps:>20.1f} | {mean_steps:>18.1f} | "
            f"{median_logged:>17.1f} | {avg_load_before:>15.2f} | {avg_load_after:>14.2f}"
        )

    best_steps, best_n_envs = max(ranking, key=lambda item: item[0])
    print()
    print(
        f"Doporucena konfigurace: N_ENVS={best_n_envs} "
        f"(median_steps_per_sec={best_steps:.1f})"
    )
    print("Tip: pokud na stroji bezne bezi i dalsi tezke ulohy, ber jako druhou volbu o 1 nizsi N_ENVS.")


def main():
    args = parse_args()
    run_dir = args.benchmark_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Benchmark budu merit pres wall-clock cas a load average. "
        "Nesaham na hlavni model ani replay buffer."
    )
    print(
        f"CPU cores={cpu_count()}, configs={args.configs}, repeats={args.repeats}, "
        f"timesteps={args.timesteps}, vec_env={args.vec_env}, device={args.device}"
    )

    results: list[RunResult] = []
    for n_envs in args.configs:
        for repeat in range(1, args.repeats + 1):
            result = run_single_benchmark(args, run_dir, n_envs, repeat)
            results.append(result)
            if (
                repeat != args.repeats
                or n_envs != args.configs[-1]
            ) and args.cooldown_seconds > 0:
                print(f"Cooldown {args.cooldown_seconds}s...")
                time.sleep(args.cooldown_seconds)

    summarize(results)


if __name__ == "__main__":
    main()
