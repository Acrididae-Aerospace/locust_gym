"""Fixedwing Level Flight Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.gym_envs.fixedwing_envs.fixedwing_base_env import FixedwingBaseEnv


class FixedwingLevelFlightEnv(FixedwingBaseEnv):
    """Fixedwing Level Flight Environment.

    Actions are roll, pitch, yaw, thrust commands.
    The goal is to reach an altitude of 100 feet and maintain level flight for 5 seconds.

    Args:
        target_altitude (float): target altitude in meters (default 30.48 m = 100 ft).
        level_flight_duration (float): duration to maintain level flight in seconds.
        min_starting_height (float): minimum starting height in meters.
        max_starting_height (float): maximum starting height in meters.
        min_starting_velocity (float): minimum starting velocity in m/s.
        max_starting_velocity (float): maximum starting velocity in m/s.
        min_ground_angle (float): minimum pitch angle when below altitude_angle_threshold (degrees).
        max_ground_angle (float): maximum pitch angle when below altitude_angle_threshold (degrees).
        min_air_angle (float): minimum pitch angle when above altitude_angle_threshold (degrees).
        max_air_angle (float): maximum pitch angle when above altitude_angle_threshold (degrees).
        altitude_angle_threshold (float): height threshold for angle interpolation (meters).
        min_roll_angle (float): minimum roll angle (degrees).
        max_roll_angle (float): maximum roll angle (degrees).
        level_roll_threshold (float): maximum roll angle considered level (degrees).
        level_pitch_threshold (float): maximum pitch angle considered level (degrees).
        level_yaw_rate_threshold (float): maximum yaw rate considered level (rad/s).
        flight_mode (int): The flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution
    """

    def __init__(
        self,
        target_altitude: float = 60.96,  # in meters
        level_flight_duration: float = 2.0,
        min_starting_height: float = 2,
        max_starting_height: float =  2.5, #15.24, # in meters
        min_starting_velocity: float = 8.0,
        max_starting_velocity: float = 18.0,
        min_ground_angle: float = 215.0,
        max_ground_angle: float = 270.0,
        min_air_angle: float = 180,
        max_air_angle: float = 270.0,
        altitude_angle_threshold: float = 100,  # in meters not used rn
        min_roll_angle: float = 0.0,
        max_roll_angle: float = 360.0,
        level_roll_threshold: float = 5.0,
        level_pitch_threshold: float = 5.0,
        level_yaw_rate_threshold: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 300.0,
        max_duration_seconds: float = 120.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__."""
        super().__init__(
            # Start position will be overridden in reset
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            # Start orientation will be overridden in reset
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # Define level flight parameters
        self.target_altitude = target_altitude
        self.level_flight_duration = level_flight_duration
        self.time_at_level_flight = 0.0
        self.time_per_step = 1.0 / agent_hz

        # Define starting condition parameters
        self.min_starting_height = min_starting_height
        self.max_starting_height = max_starting_height
        self.min_starting_velocity = min_starting_velocity
        self.max_starting_velocity = max_starting_velocity
        self.min_ground_angle = min_ground_angle
        self.max_ground_angle = max_ground_angle
        self.min_air_angle = min_air_angle
        self.max_air_angle = max_air_angle
        self.altitude_angle_threshold = altitude_angle_threshold
        self.min_roll_angle = min_roll_angle
        self.max_roll_angle = max_roll_angle

        # Define what constitutes "level flight"
        self.level_roll_threshold = level_roll_threshold
        self.level_pitch_threshold = level_pitch_threshold
        self.level_yaw_rate_threshold = level_yaw_rate_threshold

        # Store initial velocity for reset
        self.starting_velocity = None

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "altitude_info": spaces.Box(
                    low=np.array([-np.inf, 0.0], dtype=np.float64),
                    high=np.array([np.inf, np.inf], dtype=np.float64),
                    shape=(2,),
                    dtype=np.float64,
                ),
            }
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "altitude_info"], np.ndarray], dict]:
        """Reset the environment with randomized starting conditions."""
        # Reset the level flight timer
        self.time_at_level_flight = 0.0

        # Generate random starting height
        starting_height = self.np_random.uniform(
            self.min_starting_height, self.max_starting_height
        )

        # Calculate starting pitch angle based on height
        if starting_height > self.altitude_angle_threshold:
            # Interpolate angle based on current height
            t = starting_height / (self.max_starting_height - self.altitude_angle_threshold)
            min_angle = np.radians(self.min_ground_angle)
            max_angle = np.radians(self.max_ground_angle)
            min_air = np.radians(self.min_air_angle)
            max_air = np.radians(self.max_air_angle)

            # Interpolate from ground angle to air angle
            min_pitch = min_angle * (1 - t) + min_air * t
            max_pitch = max_angle * (1 - t) + max_air * t
        else: # only ground
            min_pitch = np.radians(self.min_ground_angle)
            max_pitch = np.radians(self.max_ground_angle)

        pitch_angle = self.np_random.uniform(min_pitch, max_pitch)

        # Generate random roll angle
        roll_angle = self.np_random.uniform(
            np.radians(self.min_roll_angle), np.radians(self.max_roll_angle)
        )

        # Generate random yaw angle
        yaw_angle = self.np_random.uniform(-np.pi, np.pi)

        # Set starting position and orientation
        self.start_pos = np.array([[0.0, 0.0, starting_height]])
        self.start_orn = np.array([[roll_angle, pitch_angle, yaw_angle]])

        # Generate random starting velocity
        starting_velocity = self.np_random.uniform(
            self.min_starting_velocity, self.max_starting_velocity
        )

        # Calculate velocity vector based on orientation
        # Create rotation matrix from Euler angles (roll, pitch, yaw)
        # This ensures the velocity aligns with the aircraft's orientation

        # First, create rotation matrices for each axis
        # Roll (rotation around X axis)
        R_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll_angle), -np.sin(roll_angle)],
                [0, np.sin(roll_angle), np.cos(roll_angle)],
            ]
        )

        # Pitch (rotation around Y axis)
        R_pitch = np.array(
            [
                [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                [0, 1, 0],
                [-np.sin(pitch_angle), 0, np.cos(pitch_angle)],
            ]
        )

        # Yaw (rotation around Z axis)
        R_yaw = np.array(
            [
                [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                [0, 0, 1],
            ]
        )

        # Combine rotations (order: yaw -> pitch -> roll)
        R = R_yaw @ R_pitch @ R_roll

        # Create forward vector and apply rotation
        forward_vector = np.array([1.0, 0.0, 0.0])  # Forward along X-axis
        velocity_direction = R @ forward_vector

        # Normalize to unit vector and scale by velocity
        velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
        self.starting_velocity = velocity_direction * starting_velocity

        # Begin the environment reset
        drone_options = {
            "starting_velocity": self.starting_velocity,  # Pass initial velocity as drone option
        }
        super().begin_reset(seed, options, drone_options)

        # Initialize info dictionary
        self.info["level_flight_time"] = 0.0
        self.info["is_level_flight"] = False
        self.info["reached_target_altitude"] = False
        self.info["completed_task"] = False

        # Finish the reset
        super().end_reset()

        return self.state, self.info

    def compute_state(self) -> None:
        """Compute the current state of the environment."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # Calculate altitude and difference from target
        current_altitude = lin_pos[2]
        altitude_error = self.target_altitude - current_altitude

        # Combine everything for the state
        new_state: dict[Literal["attitude", "altitude_info"], np.ndarray] = dict()

        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )

        # Add altitude information
        new_state["altitude_info"] = np.array(
            [
                altitude_error,  # Error from target altitude
                current_altitude,  # Current altitude
            ]
        )

        self.state = new_state

    def compute_term_trunc_reward(self) -> None:
        """Compute termination conditions, truncation, and continuous rewards."""
        super().compute_base_term_trunc_reward()

        # Get current state information
        ang_vel, ang_pos, _, lin_pos, _ = super().compute_attitude()

        # Altitude
        current_altitude = lin_pos[2]
        altitude_error = abs(current_altitude - self.target_altitude)

        # Altitude reward with smoother scaling
        altitude_reward = 0
        if current_altitude < self.target_altitude:
            # Gradually increase reward as it approaches target altitude
            altitude_reward = 10 * (current_altitude / self.target_altitude)
        elif current_altitude > self.target_altitude + 1:
            # Increasingly negative reward for being too high
            altitude_reward = -5 * (altitude_error / self.target_altitude)
        else: # in window
            altitude_reward = 15

        # Continuous levelness reward
        # Extract roll, pitch from euler angles
        roll, pitch, _ = ang_pos
        _, _, yaw_rate = ang_vel

        # Convert to degrees for more intuitive scaling
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)

        # Compute levelness as a continuous score
        # Lower deviation from zero means more level
        roll_levelness = 1 - min(1, abs(roll_deg - 180) / self.level_roll_threshold)
        pitch_levelness = 1 - min(1, abs(pitch_deg - 180) / self.level_pitch_threshold)
        yaw_stability = 1 - min(1, abs(yaw_rate) / self.level_yaw_rate_threshold)

        # Combined levelness score with some weighting
        levelness_reward = 10 * (roll_levelness * pitch_levelness * yaw_stability)

        # Time penalty with exponential increase
        time_penalty = -0.01 * (1 + self.step_count / self.max_steps)

        # Distance penalty
        # Calculate horizontal distance from origin (x, y)
        horizontal_dist = np.linalg.norm(lin_pos[:2])
        
        # Distance penalty
        if horizontal_dist <= 50:
            distance_penalty = 0
        else:
            # Squared error beyond 50m
            distance_penalty = -((horizontal_dist - 50) ** 2) / 1000

        # Combine rewards
        self.reward = (
            altitude_reward + 
            levelness_reward + 
            time_penalty + 
            distance_penalty
        )

        # Update info dictionary
        self.info["reached_target_altitude"] = current_altitude >= self.target_altitude
        self.info["roll_levelness"] = roll_levelness
        self.info["pitch_levelness"] = pitch_levelness
        self.info["yaw_stability"] = yaw_stability
        self.info["horizontal_distance"] = horizontal_dist

        # Check for successful completion
        self.time_at_level_flight += self.time_per_step
        if (current_altitude >= self.target_altitude and
            roll_levelness > 0.9 and
            pitch_levelness > 0.9 and
            yaw_stability > 0.9 and
            self.time_at_level_flight >= self.level_flight_duration):
            self.reward += 1000.0  # Large bonus for completing the task
            self.termination = True
            self.info["completed_task"] = True
            self.info["env_complete"] = True
        else:
            self.info["completed_task"] = False
            self.info["env_complete"] = False