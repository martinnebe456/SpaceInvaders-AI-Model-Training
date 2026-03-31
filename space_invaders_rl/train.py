"""Train or continue a DQN agent for the Space Invaders environment."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import SpaceInvadersEnv


def parse_args():
    """Parse CLI options for a resumable training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of additional timesteps to train in this run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Maximum number of steps per training episode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/space_invaders_dqn"),
    )
    parser.add_argument(
        "--replay-buffer-path",
        type=Path,
        default=Path("models/space_invaders_dqn_replay_buffer.pkl"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/space_invaders_dqn"),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--exploration-fraction", type=float, default=0.15)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--skip-check-env", action="store_true")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore any existing model and start training from scratch.",
    )
    parser.add_argument(
        "--reset-replay-buffer",
        action="store_true",
        help="Load the existing model but ignore the saved replay buffer.",
    )
    return parser.parse_args()


def ensure_zip_suffix(path: Path) -> Path:
    """Normalize model paths so Stable-Baselines always saves a .zip file."""
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def resolve_device(device_name: str) -> str:
    """Resolve the requested training device, falling back from auto mode."""
    if device_name != "auto":
        return device_name

    # Prefer Apple Metal on supported Macs, then CUDA, then plain CPU.
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def build_env(args):
    """Create the vectorized training environment from CLI settings."""
    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    # "spawn" is the safest start method across macOS and Windows subprocess envs.
    vec_env_kwargs = {"start_method": "spawn"} if args.vec_env == "subproc" else None
    return make_vec_env(
        SpaceInvadersEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={
            "render_mode": None,
            "max_steps": args.max_steps,
            "single_level_mode": True,
        },
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )


def build_new_model(args, env, resolved_device: str):
    """Construct a fresh DQN model with the current hyperparameter set."""
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [256, 256, 256]},
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device=resolved_device,
    )


def main():
    """Run one training session and persist both the model and replay buffer."""
    args = parse_args()
    model_path = ensure_zip_suffix(args.model_path)
    resolved_device = resolve_device(args.device)

    # Create output directories up front so later save operations cannot fail
    # just because the path tree does not exist yet.
    args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_check_env:
        # Validate a single non-vectorized env once before training. This catches
        # Gym API mistakes early without paying the cost every rollout.
        validation_env = SpaceInvadersEnv(max_steps=args.max_steps)
        try:
            check_env(validation_env, warn=True)
        finally:
            validation_env.close()

    env = build_env(args)
    print(
        "Training configuration:"
        f" device={resolved_device}, n_envs={args.n_envs},"
        f" vec_env={args.vec_env}, timesteps={args.timesteps},"
        f" max_steps={args.max_steps}"
    )

    try:
        should_resume = model_path.exists() and not args.fresh_start
        if should_resume:
            # Resume weights from the latest saved model unless the user
            # explicitly requested a clean restart.
            print(f"Loading existing model from {model_path}")
            model = DQN.load(
                str(model_path),
                env=env,
                tensorboard_log=str(args.tensorboard_log),
                device=resolved_device,
            )

            if args.reset_replay_buffer:
                print("Replay buffer was intentionally skipped (--reset-replay-buffer).")
            elif args.replay_buffer_path.exists():
                print(f"Loading replay buffer from {args.replay_buffer_path}")
                try:
                    # Restoring the replay buffer lets the optimizer keep learning
                    # from previously collected experience instead of warming up again.
                    model.load_replay_buffer(str(args.replay_buffer_path))
                except Exception as exc:
                    print(
                        "Replay buffer could not be loaded "
                        f"(likely because the env configuration changed): {exc}"
                    )
                    print("Continuing without a loaded replay buffer.")
            else:
                print("Replay buffer was not found, continuing without it.")

            # Keep the global timestep counter increasing across resumed runs so
            # TensorBoard and exploration schedules reflect the full training history.
            reset_num_timesteps = False
        else:
            print("Starting a new training run from scratch.")
            model = build_new_model(args, env, resolved_device)
            reset_num_timesteps = True

        # Train for one additional chunk of timesteps. The shell scripts can call
        # this repeatedly to build up multi-hour or multi-day training runs.
        model.learn(
            total_timesteps=args.timesteps,
            log_interval=10,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="continued_dqn",
        )

        model.save(str(model_path))
        model.save_replay_buffer(str(args.replay_buffer_path))

        # Save timestamped checkpoints as immutable recovery points in case the
        # next training cycle regresses or a long run gets interrupted later.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_model = args.checkpoint_dir / f"space_invaders_dqn_{stamp}"
        checkpoint_buffer = (
            args.checkpoint_dir / f"space_invaders_dqn_{stamp}_replay_buffer.pkl"
        )
        model.save(str(checkpoint_model))
        model.save_replay_buffer(str(checkpoint_buffer))

        print(f"Model saved to {model_path}")
        print(f"Replay buffer saved to {args.replay_buffer_path}")
        print(f"Checkpoint saved to {checkpoint_model}.zip")
    finally:
        try:
            env.close()
        except EOFError:
            # Subprocess workers may already be gone after an earlier failure.
            print("Env workers are already closed, skipping the extra close().")


if __name__ == "__main__":
    main()
