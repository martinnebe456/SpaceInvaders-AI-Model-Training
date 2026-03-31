"""Evaluate a trained DQN agent on the Space Invaders environment."""

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from .env import SpaceInvadersEnv


def parse_args():
    """Parse CLI options for silent or visual evaluation runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/space_invaders_dqn.zip"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "none"],
        default="none",
    )
    parser.add_argument(
        "--human-max-steps",
        type=int,
        default=20000,
        help="Recommended shorter limit for real-time playback in human mode.",
    )
    parser.add_argument(
        "--full-game",
        action="store_true",
        help="Keep playing past the first cleared level instead of stopping the episode.",
    )
    return parser.parse_args()


def main():
    """Run one evaluation pass and print per-episode and aggregate metrics."""
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    max_steps = args.max_steps

    # Human rendering runs in real time, so a lower default step limit keeps
    # visual playback from taking too long when the caller forgets to override it.
    if render_mode == "human" and args.max_steps == 50000:
        max_steps = args.human_max_steps

    env = SpaceInvadersEnv(
        render_mode=render_mode,
        max_steps=max_steps,
        seed=args.seed,
        single_level_mode=not args.full_game,
    )
    model = DQN.load(str(args.model_path))

    try:
        if render_mode == "human":
            print(
                f"Human rendering runs in real time at 60 FPS, "
                f"max_steps={max_steps}, full_game={args.full_game}."
            )

        total_reward = 0.0
        total_score = 0
        victories = 0
        truncations = 0
        total_player_shots_fired = 0
        total_player_shots_missed = 0
        total_player_targeted_shots = 0
        total_player_blocked_shots = 0
        total_player_bunker_hits = 0

        for episode in range(args.episodes):
            # Use a deterministic but distinct seed per episode so repeated
            # evaluations are reproducible without replaying the exact same level.
            episode_seed = args.seed + episode
            observation, info = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if render_mode == "rgb_array":
                    # Force frame generation even when the caller is not showing
                    # a live window, which is useful for recording or debugging.
                    env.render()

            total_reward += episode_reward
            total_score += info["score"]
            victories += int(info["victory"])
            truncations += int(truncated)
            total_player_shots_fired += info["player_shots_fired"]
            total_player_shots_missed += info["player_shots_missed"]
            total_player_targeted_shots += info["player_targeted_shots"]
            total_player_blocked_shots += info["player_blocked_shots"]
            total_player_bunker_hits += info["player_bunker_hits"]
            print(
                f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                f"score={info['score']}, lives={info['lives']}, "
                f"victory={info['victory']}, truncated={truncated}, "
                f"shots={info['player_shots_fired']}, "
                f"missed={info['player_shots_missed']}, "
                f"targeted={info['player_targeted_shots']}, "
                f"blocked={info['player_blocked_shots']}, "
                f"bunker_hits={info['player_bunker_hits']}"
            )

        # Aggregate metrics make it easier to compare checkpoints than raw
        # per-episode logs, especially when outcomes are still noisy.
        print(
            f"Summary: mean_reward={total_reward / args.episodes:.2f}, "
            f"mean_score={total_score / args.episodes:.2f}, "
            f"victories={victories}/{args.episodes}, "
            f"truncated={truncations}/{args.episodes}, "
            f"mean_shots={total_player_shots_fired / args.episodes:.2f}, "
            f"mean_missed={total_player_shots_missed / args.episodes:.2f}, "
            f"mean_targeted={total_player_targeted_shots / args.episodes:.2f}, "
            f"mean_blocked={total_player_blocked_shots / args.episodes:.2f}, "
            f"mean_bunker_hits={total_player_bunker_hits / args.episodes:.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
