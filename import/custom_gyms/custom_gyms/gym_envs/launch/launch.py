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
        level_angle_rate_threshold (float): maximum rate considered level (rad/s).
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
        target_altitude: float = 30.48,  # in meters
        level_flight_duration: float = 1.5,
        min_starting_height: float = 2,
        max_starting_height: float = 2.5,  # 15.24, # in meters
        min_starting_velocity: float = 8.0,
        max_starting_velocity: float = 18.0,
        min_ground_angle: float = 215.0,
        max_ground_angle: float = 270.0,
        min_air_angle: float = 180,
        max_air_angle: float = 270.0,
        altitude_angle_threshold: float = 100,  # in meters not used rn
        min_roll_angle: float = 0.0,
        max_roll_angle: float = 360.0,
        level_roll_threshold: float = 10.0,
        level_pitch_threshold: float = 10.0,
        level_angle_rate_threshold: float = 0.2,
        flight_mode: int = 0,
        flight_dome_size: float = 150.0,
        max_duration_seconds: float = 20.0,
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
        self.agent_hz = agent_hz
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
        self.level_angle_rate_threshold = level_angle_rate_threshold

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
            t = starting_height / (
                self.max_starting_height - self.altitude_angle_threshold
            )
            min_angle = np.radians(self.min_ground_angle)
            max_angle = np.radians(self.max_ground_angle)
            min_air = np.radians(self.min_air_angle)
            max_air = np.radians(self.max_air_angle)

            # Interpolate from ground angle to air angle
            min_pitch = min_angle * (1 - t) + min_air * t
            max_pitch = max_angle * (1 - t) + max_air * t
        else:  # only ground
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
        """
        A revised terminal/truncation reward function that uses smoother, 
        potential-based shaping and moderate bonuses to encourage:
        1. Climbing toward a target altitude.
        2. Maintaining near-level orientation (roll/pitch).
        3. Keeping angular rates small.
        4. Staying alive longer instead of crashing quickly.
        5. Providing a final moderate bonus if stable flight is achieved.
        """
        # Always call the base reward logic (handles collisions, timeouts, etc.)
        super().compute_base_term_trunc_reward()

        # If base logic already terminated or truncated the episode, don't proceed
        if self.termination or self.truncation:
            return

        # Retrieve current state information from a helper
        ang_vel, ang_pos, _, lin_pos, _ = super().compute_attitude()
        roll_rate, pitch_rate, yaw_rate = ang_vel
        roll, pitch, yaw = ang_pos
        current_altitude = lin_pos[2]

        # ----------------------------
        # 1. Time-Alive Reward
        # ----------------------------
        # A small positive reward for every step the drone remains in flight.
        # Helps discourage "crash early" policies.
        time_alive_reward = 0.01

        # ----------------------------
        # 2. Altitude Shaping
        # ----------------------------
        # Encourage being close to target_altitude with a smooth exponential.
        # This will be ~1.0 if at target altitude, and drop off as error grows.
        altitude_error = abs(current_altitude - self.target_altitude)
        altitude_shaping = np.exp(-0.1 * altitude_error)  # in range (0, 1]

        # ----------------------------
        # 3. Orientation Shaping
        # ----------------------------
        # Suppose "level" is near 180 deg roll & 180 deg pitch (from your original code).
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        # print(roll_deg)
        roll_error = roll_deg
        pitch_error = pitch_deg
        # Exponential shaping that rewards small deviation from "level."
        orientation_shaping = np.exp(-0.01 * (roll_error**2 + pitch_error**2))

        # ----------------------------
        # 4. Angular Rate Penalty
        # ----------------------------
        # Soft penalty for large angular velocities (roll_rate, pitch_rate, yaw_rate).
        # Negative sign because large rates reduce the reward.
        angular_rate_penalty = -0.1 * (roll_rate**2 + pitch_rate**2 + yaw_rate**2)

        # ----------------------------
        # 5. Consecutive Stability
        # ----------------------------
        # If the drone is "reasonably level" and "reasonably stable," increment
        # a counter. Once it stays stable for a while, we give a final bonus.
        # (Adjust thresholds to your preference.)
        altitude_within_band = (abs(current_altitude - self.target_altitude) < 5.0)
        stable_roll = abs(roll_error) < self.level_roll_threshold   # within 10 deg of 180
        # print(roll_error)
        stable_pitch = abs(pitch_error) < self.level_pitch_threshold
        stable_rates = (abs(roll_rate) < self.level_angle_rate_threshold and
                        abs(pitch_rate) < self.level_angle_rate_threshold and
                        abs(yaw_rate)   < self.level_angle_rate_threshold)

        if stable_roll and stable_pitch and stable_rates:
            self.consecutive_stable_frames += 1
        else:
            self.consecutive_stable_frames = 0

        # A small incremental shaping for maintaining stability across frames
        stability_progression_reward = 0.1 * self.consecutive_stable_frames

        # ----------------------------
        # 6. Summation of Shaping Terms
        # ----------------------------
        # Add all shaping and penalties to the agent's reward. Keep magnitudes moderate.
        step_reward = (
            time_alive_reward
            + altitude_shaping
            + orientation_shaping
            + angular_rate_penalty
            + stability_progression_reward
        )
        if(altitude_within_band):
            step_reward += (
                + orientation_shaping
                + angular_rate_penalty
                + stability_progression_reward
            )
            step_reward *= 2
        self.reward += step_reward

        # ----------------------------
        # 7. Check for "Successful Flight" Condition
        # ----------------------------
        # If altitude is within some band of the target and we've maintained
        # stable orientation for enough frames, we consider the task completed.
        # Give a moderate bonus and terminate.
        # (Adjust thresholds as needed.)
        stable_time_requirement = self.level_flight_duration / self.time_per_step

        if altitude_within_band:
            self.info["reached_target_altitude"] = True
        self.info["level_flight_time"] = max(self.info["level_flight_time"], self.consecutive_stable_frames / self.agent_hz)

        if altitude_within_band \
        and stable_roll and stable_pitch and stable_rates \
        and self.consecutive_stable_frames >= stable_time_requirement:
            # Final moderate success bonus
            self.reward += 10000.0
            self.termination = True
            self.info["completed_task"] = True
            self.info["env_complete"] = True
        else:
            self.info["completed_task"] = False
            self.info["env_complete"] = False

        # ----------------------------
        # 8. Diagnostic Info
        # ----------------------------
        # You can log intermediate values for debugging/analysis
        self.info.update({
            "altitude_error": altitude_error,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "altitude_shaping": altitude_shaping,
            "orientation_shaping": orientation_shaping,
            "angular_rate_penalty": angular_rate_penalty,
            "stability_progression_reward": stability_progression_reward,
            "consecutive_stable_frames_target": stable_time_requirement,
            "consecutive_stable_frames": self.consecutive_stable_frames,
            "stable_roll": stable_roll,
            "stable_pitch": stable_pitch,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "stable_rates": stable_rates,
        })