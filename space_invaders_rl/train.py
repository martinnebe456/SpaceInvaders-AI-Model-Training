"""Prubezny DQN trening pro Space Invaders se schopnosti navazat."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Pocet dalsich kroku, ktere se maji natrenovat v tomto behu.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Limit kroku jedne epizody pri treningu.",
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
        help="Ignoruje existujici model a zacne trenink od nuly.",
    )
    parser.add_argument(
        "--reset-replay-buffer",
        action="store_true",
        help="Nacte existujici model, ale ignoruje ulozeny replay buffer.",
    )
    return parser.parse_args()


def ensure_zip_suffix(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def build_env(args):
    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
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
    args = parse_args()
    model_path = ensure_zip_suffix(args.model_path)
    resolved_device = resolve_device(args.device)
    args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_check_env:
        validation_env = SpaceInvadersEnv(max_steps=args.max_steps)
        try:
            check_env(validation_env, warn=True)
        finally:
            validation_env.close()

    env = build_env(args)
    print(
        "Konfigurace treningu:"
        f" device={resolved_device}, n_envs={args.n_envs},"
        f" vec_env={args.vec_env}, timesteps={args.timesteps},"
        f" max_steps={args.max_steps}"
    )

    try:
        should_resume = model_path.exists() and not args.fresh_start
        if should_resume:
            print(f"Nacitam existujici model z {model_path}")
            model = DQN.load(
                str(model_path),
                env=env,
                tensorboard_log=str(args.tensorboard_log),
                device=resolved_device,
            )

            if args.reset_replay_buffer:
                print("Replay buffer byl zamerne preskocen (--reset-replay-buffer).")
            elif args.replay_buffer_path.exists():
                print(f"Nacitam replay buffer z {args.replay_buffer_path}")
                try:
                    model.load_replay_buffer(str(args.replay_buffer_path))
                except Exception as exc:
                    print(
                        "Replay buffer se nepodarilo nacist "
                        f"(pravdepodobne jina konfigurace envu): {exc}"
                    )
                    print("Pokračuji bez nacteneho replay bufferu.")
            else:
                print("Replay buffer nebyl nalezen, pokracuji bez nej.")

            reset_num_timesteps = False
        else:
            print("Zacinam novy trening od nuly.")
            model = build_new_model(args, env, resolved_device)
            reset_num_timesteps = True

        model.learn(
            total_timesteps=args.timesteps,
            log_interval=10,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="continued_dqn",
        )

        model.save(str(model_path))
        model.save_replay_buffer(str(args.replay_buffer_path))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_model = args.checkpoint_dir / f"space_invaders_dqn_{stamp}"
        checkpoint_buffer = (
            args.checkpoint_dir / f"space_invaders_dqn_{stamp}_replay_buffer.pkl"
        )
        model.save(str(checkpoint_model))
        model.save_replay_buffer(str(checkpoint_buffer))

        print(f"Model ulozen do {model_path}")
        print(f"Replay buffer ulozen do {args.replay_buffer_path}")
        print(f"Checkpoint ulozen do {checkpoint_model}.zip")
    finally:
        try:
            env.close()
        except EOFError:
            print("Env workers uz byly ukonceny, preskakuji dodatecne close().")


if __name__ == "__main__":
    main()
