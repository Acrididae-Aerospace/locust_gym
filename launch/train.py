import gymnasium as gym
import PyFlyt.gym_envs
import custom_gyms.gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import os


class DroneResultsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_trials = 0
        self.level_flight_times = []
        self.save_freq = 100
        # store results for each drone
        self.num_success = 0
        self.num_crash = 0
        self.num_timed_out = 0
        self.num_bounds = 0
        self.reached_target_altitude = 0
        self.recorded = []

    def _on_step(self) -> bool:
        for env_idx in range(self.model.n_envs):
            # Get the terminal observation of the just finished trajectory
            if self.locals["dones"][env_idx] and not env_idx in self.recorded:
                self.total_trials += 1
                self.level_flight_times.append(
                    self.locals["infos"][env_idx]["level_flight_time"]
                )
                self.recorded.append(env_idx)
                # print(self.locals["infos"][env_idx])
                # add log
                if self.locals["infos"][env_idx]["out_of_bounds"]:
                    self.num_bounds += 1
                if self.locals["infos"][env_idx]["reached_target_altitude"]:
                    self.reached_target_altitude += 1
                if self.locals["infos"][env_idx]["collision"]:
                    self.num_crash += 1
                if self.locals["infos"][env_idx]["timed_out"]:
                    self.num_timed_out += 1
                if self.locals["infos"][env_idx]["completed_task"]:
                    self.num_success += 1
            elif not self.locals["dones"][env_idx] and env_idx in self.recorded:
                self.recorded.remove(env_idx)
        # Check save freq
        if (
            self.num_success + self.num_crash + self.num_timed_out + self.num_bounds
            >= self.save_freq
        ):
            # log
            self.logger.record("final_metrics/total_trials", self.total_trials)
            # outcomes
            # print(self.num_crash)
            self.logger.record("final_metrics/crashes", self.num_crash)
            self.logger.record(
                "final_metrics/reached_target_altitude", self.reached_target_altitude
            )
            self.logger.record("final_metrics/out_of_bounds", self.num_bounds)
            self.logger.record("final_metrics/timeouts", self.num_timed_out)
            self.logger.record("final_metrics/successes", self.num_success)
            # metrics
            if len(self.level_flight_times) == 0:
                self.logger.record("final_metrics/level_flight_time_mean", 0)
            else:
                self.logger.record(
                    "final_metrics/level_flight_time_mean",
                    sum(self.level_flight_times) / len(self.level_flight_times),
                )
            # clear
            self.level_flight_times = []
            self.num_success = 0
            self.num_crash = 0
            self.num_timed_out = 0
            self.num_bounds = 0
            self.reached_target_altitude = 0
        return True


# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("final_models", exist_ok=True)

wandb.init(
    project="pyflyt-launch-training",
    name="PPO_FIXED_WING_LAUNCH",
    sync_tensorboard=True,
    monitor_gym=False,
)


def train_model(
    timesteps=100000000000,
    save_path="./final_models/ppo_drone_launch_final",
    checkpoint_interval=10000,
):
    """
    Trains PPO on the PyFlyt fixed wing launch env with periodic checkpoints.
    Args:
        timesteps: Total timesteps for training
        save_path: Path to save the final model
        checkpoint_interval: Frequency (in timesteps) for saving model checkpoints
    """
    train_env = make_vec_env("Launch/Drone-launch-v0", n_envs=4)
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        device="cpu",
    )
        # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path="./checkpoints/",
        name_prefix="ppo_drone-launch",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Combine with WandB callback
    callbacks = [
        WandbCallback(),
        checkpoint_callback,
        DroneResultsCallback(),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    train_env.close()


if __name__ == "__main__":
    # Train with checkpoints every 50,000 steps
    train_model(timesteps=1000000000, checkpoint_interval=5000)
