"""Registers PyFlyt environments into Gymnasium."""

from gymnasium.envs.registration import register

# Fixed winged Launch Envs
register(
    id="Launch/Drone-launch-v0",
    entry_point="custom_gyms.gym_envs.launch.launch:FixedwingWaypointsEnv",
)