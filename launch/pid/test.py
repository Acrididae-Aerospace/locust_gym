import gymnasium as gym
import custom_gyms.gym_envs
import os
import argparse
import sys
import numpy as np

# Import the PID controller from the previous artifact
from pid_controller import create_pid_controller


def test_pid_controller(controller, num_episodes=50):
    """
    Tests the PID controller with rendering for multiple episodes.

    Args:
        controller (PIDController): PID controller to test
        num_episodes (int): Number of episodes to test the controller
    """
    try:
        test_env = gym.make("Launch/Drone-launch-v0", render_mode="human")

        print(f"\nTesting PID Controller")
        print(f"Running for {num_episodes} episodes...\n")

        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                # Use the PID controller to predict actions
                action, _ = controller.predict(obs)

                # Optional: Print observation and action for debugging
                # print("Observation:", obs)
                print("Action:", action)

                # Step the environment
                obs, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1

            print(
                f"Episode {episode + 1}/{num_episodes} - Steps: {steps} - Total Reward: {episode_reward:.2f}"
            )

        test_env.close()
        return True

    except Exception as e:
        print(f"Error during PID controller testing: {str(e)}")
        return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Test a PID controller in the PyFlyt environment."
    )

    parser.add_argument(
        "--kp",
        type=float,
        default=1.0,
        help="Proportional gain (default: 1.0)",
    )

    parser.add_argument(
        "--ki",
        type=float,
        default=0.1,
        help="Integral gain (default: 0.1)",
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=0.01,
        help="Derivative gain (default: 0.01)",
    )

    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=50,
        help="Number of episodes to test the controller (default: 50)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create the environment to initialize the controller
    env = gym.make("Launch/Drone-launch-v0")

    # Create PID controller with specified gains
    pid_controller = create_pid_controller(env, kp=args.kp, ki=args.ki, kd=args.kd)

    # Test the PID controller
    success = test_pid_controller(pid_controller, args.episodes)

    if not success:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
