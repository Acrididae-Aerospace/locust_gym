import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import os

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

wandb.init(
    project="pyflyt-training",
    name="PPO_QUAD_POLE_BALANCING",
    sync_tensorboard=True,
    monitor_gym=False,
)


def train_model(
    timesteps=1000000000,
    save_path="./final_models/ppo_quad_pole_balance_final",
    checkpoint_interval=10000,
):
    """
    Trains PPO on the PyFlyt QuadX-Pole-Balance environment with periodic checkpoints.
    Args:
        timesteps: Total timesteps for training
        save_path: Path to save the final model
        checkpoint_interval: Frequency (in timesteps) for saving model checkpoints
    """
    train_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v4", n_envs=4)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        device="cpu",
    )
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path="./checkpoints/",
        name_prefix="ppo_quad_pole_balance_steps",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    # Combine with WandB callback
    callbacks = [WandbCallback(), checkpoint_callback]

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    train_env.close()


def test_model(load_path="ppo_quad_pole_balance", num_episodes=50):
    """Tests the trained PPO model with rendering for multiple episodes."""
    model = PPO.load(load_path, device="cpu")
    test_env = gym.make("PyFlyt/QuadX-Pole-Balance-v4", render_mode="human")

    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(
            f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}"
        )
    test_env.close()


if __name__ == "__main__":
    # Train with checkpoints every 50,000 steps
    # train_model(timesteps=1000000000, checkpoint_interval=50000)
    # Test model
    test_model("./checkpoints/ppo_quad_pole_balance_steps_5400000_steps") # Test a specific checkpoint
    # test_model("final_models/ppo_quad_pole_balance_final")  # Test final model
    # Close WandB
    wandb.finish()
