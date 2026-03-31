"""Benchmark different training configurations for Space Invaders RL."""

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
    """Store the measured outcome of one benchmarked training run."""
    n_envs: int
    repeat: int
    elapsed_seconds: float
    steps_per_second: float
    last_logged_fps: float | None
    load_before: tuple[float, float, float]
    load_after: tuple[float, float, float]
    log_path: Path


def parse_args():
    """Parse CLI options for isolated training-speed benchmark runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=[1, 4, 6, 8, 10],
        help="List of n_envs values to benchmark.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=80_000,
        help="Number of timesteps for each measured run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many times to repeat each configuration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Maximum number of steps per episode during the benchmark.",
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
        help="Pause between runs so the machine can cool down a bit.",
    )
    parser.add_argument(
        "--max-load-ratio",
        type=float,
        default=0.75,
        help="If 1-minute loadavg / CPU count is above this threshold, wait before running.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=20,
        help="How long to wait between load checks.",
    )
    parser.add_argument(
        "--max-waits",
        type=int,
        default=6,
        help="Maximum number of times to wait for the system load to drop.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("runs/benchmark"),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the temporary models and replay buffers produced by the benchmark.",
    )
    return parser.parse_args()


def cpu_count() -> int:
    """Return the number of logical CPUs, defaulting to 1 if unknown."""
    return os.cpu_count() or 1


def get_loadavg() -> tuple[float, float, float]:
    """Return the OS load average when supported, otherwise zeros."""
    if not hasattr(os, "getloadavg"):
        return (0.0, 0.0, 0.0)

    try:
        return os.getloadavg()
    except OSError:
        return (0.0, 0.0, 0.0)


def load_ratio(load: tuple[float, float, float]) -> float:
    """Normalize the 1-minute load average by the number of CPU cores."""
    return load[0] / cpu_count()


def wait_for_reasonable_load(args) -> tuple[float, float, float]:
    """Delay the benchmark if the machine is already busy with other work."""
    current_load = get_loadavg()
    if current_load == (0.0, 0.0, 0.0):
        return current_load

    attempts = 0
    while load_ratio(current_load) > args.max_load_ratio and attempts < args.max_waits:
        ratio = load_ratio(current_load)
        print(
            "The system is currently fairly busy: "
            f"load1={current_load[0]:.2f}, ratio={ratio:.2f}. "
            f"Waiting {args.wait_seconds}s..."
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
    """Build the exact training command used for one isolated benchmark run."""
    vec_env = args.vec_env
    if vec_env == "auto":
        # A single env is cheapest in-process, while multiple envs are usually
        # faster in subprocesses.
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
    """Run one benchmarked training process and collect timing data."""
    run_name = f"env{n_envs}_run{repeat}"
    log_path = run_dir / f"{run_name}.log"
    before_load = wait_for_reasonable_load(args)
    command = benchmark_command(args, run_dir, n_envs, repeat)

    print(
        f"Starting benchmark {run_name}: "
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
            f"Benchmark {run_name} failed. See log {log_path}"
        ) from exc

    elapsed = time.perf_counter() - start
    after_load = get_loadavg()
    log_path.write_text(output, encoding="utf-8")

    logged_fps_matches = FPS_PATTERN.findall(output)
    last_logged_fps = float(logged_fps_matches[-1]) if logged_fps_matches else None
    # Wall-clock throughput is the main metric because it includes process
    # overhead, environment stepping, and model updates together.
    steps_per_second = args.timesteps / elapsed

    print(
        f"Finished {run_name}: elapsed={elapsed:.1f}s, "
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
    """Remove per-run artifacts so benchmark runs do not fill the disk."""
    model_base = run_dir / f"model_env{n_envs}_run{repeat}"
    replay_path = run_dir / f"model_env{n_envs}_run{repeat}_replay.pkl"
    checkpoint_dir = run_dir / f"checkpoints_env{n_envs}_run{repeat}"

    for path in [model_base.with_suffix(".zip"), replay_path]:
        if path.exists():
            path.unlink()

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def summarize(results: list[RunResult]) -> None:
    """Print aggregate metrics and recommend the fastest tested configuration."""
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
        f"Recommended configuration: N_ENVS={best_n_envs} "
        f"(median_steps_per_sec={best_steps:.1f})"
    )
    print(
        "Tip: if the machine is usually busy with other heavy workloads too, "
        "consider one lower N_ENVS value as the second-best option."
    )


def main():
    """Run the full benchmark matrix and print a final recommendation."""
    args = parse_args()
    run_dir = args.benchmark_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Benchmarking via wall-clock time and load average. "
        "The main model and replay buffer are left untouched."
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
                # Cooling down between runs makes the later measurements less
                # biased by thermal throttling or leftover system load.
                print(f"Cooldown {args.cooldown_seconds}s...")
                time.sleep(args.cooldown_seconds)

    summarize(results)


if __name__ == "__main__":
    main()
