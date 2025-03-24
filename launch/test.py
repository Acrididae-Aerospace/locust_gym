import gymnasium as gym
import custom_gyms.gym_envs
from stable_baselines3 import PPO
import os
import argparse
import sys


def test_model(load_path, num_episodes=50):
    """
    Tests the trained PPO model with rendering for multiple episodes.

    Args:
        load_path (str): Path to the saved model file
        num_episodes (int): Number of episodes to test the model
    """
    try:
        if not os.path.exists(load_path):
            print(f"Error: Model file '{load_path}' does not exist.")
            return False

        model = PPO.load(load_path, device="cpu")
        test_env = gym.make("Launch/Drone-launch-v0", render_mode="human")

        print(f"\nTesting model: {load_path}")
        print(f"Running for {num_episodes} episodes...\n")

        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                # print(obs)
                # print(action)
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
        print(f"Error during model testing: {str(e)}")
        return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Test a trained PPO model in the PyFlyt environment."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="n/a",
        help="Path to the saved model file",
    )

    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=50,
        help="Number of episodes to test the model (default: 50)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available model files in checkpoints and final_models directories",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle listing available models
    if args.list:
        print("\nAvailable models:")

        # Check checkpoints directory
        if os.path.exists("./checkpoints"):
            checkpoint_models = [
                f
                for f in os.listdir("./checkpoints")
                if os.path.isfile(os.path.join("./checkpoints", f))
            ]
            if checkpoint_models:
                print("\nCheckpoint models:")
                for model in checkpoint_models:
                    print(f"  ./checkpoints/{model}")

        # Check final_models directory
        if os.path.exists("./final_models"):
            final_models = [
                f
                for f in os.listdir("./final_models")
                if os.path.isfile(os.path.join("./final_models", f))
            ]
            if final_models:
                print("\nFinal models:")
                for model in final_models:
                    print(f"  ./final_models/{model}")

        return

    # Validate model path
    if not os.path.exists(args.model):
        print("Not a valid path")
        return

    # Test the model
    success = test_model(args.model, args.episodes)

    if not success:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
