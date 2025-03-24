import numpy as np

class PIDController:
    def __init__(self, env, kp=1.0, ki=0.1, kd=0.01):
        """
        Initialize a PID Controller for a gym environment with dictionary observations

        Args:
            env (gym.Env): The gymnasium environment
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.env = env

        # Determine action space dimensions
        self.action_dim = env.action_space.shape[0]

        # PID gains (can be adjusted for each control dimension)
        self.kp = np.ones(self.action_dim) * kp
        self.ki = np.ones(self.action_dim) * ki
        self.kd = np.ones(self.action_dim) * kd

        # State tracking for integral and derivative terms
        self.integral = np.zeros(self.action_dim)
        self.prev_error = np.zeros(self.action_dim)

        # Optional: Limits for integral windup and output
        self.integral_max = 10
        self.output_min = env.action_space.low
        self.output_max = env.action_space.high
        
        # Control stages
        self.current_stage = "LEVEL_ROLL"

    def _format_state(self, observation):
        """
        Extract numerical state from potentially complex observation

        Args:
            observation (dict or np.ndarray): Environment observation

        Returns:
            np.ndarray: Numerical state for PID control
        """
        # If observation is a dictionary, extract relevant numerical state
        if isinstance(observation, dict):
            # Modify this based on your specific environment's observation structure
            # This is a placeholder - you'll need to adjust based on your actual observation keys
            if "state" in observation:
                return observation["state"]
            elif "position" in observation:
                return observation["position"]
            else:
                # If no clear state, try to extract numerical values
                state = []
                for key, value in observation.items():
                    if isinstance(value, (int, float, np.number)):
                        state.append(value)
                    elif isinstance(value, np.ndarray):
                        state.extend(value.flatten())
                return np.array(state)

        # If already a numpy array, return as-is
        return np.array(observation)

    def _extract_state(self, observation):
        """
        Extract relevant state variables from the new observation model

        Observation Structure:
        - ang_vel (vector of 3 values): Angular velocities
        - ang_pos (vector of 4 values): Angular position (quaternion)
        - lin_vel (vector of 3 values): Linear velocities
        - lin_pos (vector of 3 values): Linear positions
        - previous action (4 values)
        - auxiliary information (6 values)
        - last 2 values: altitude information

        Args:
            observation (np.ndarray): Full state observation

        Returns:
            np.ndarray: Extracted state for control
        """
        # Unpack observation sections
        ang_vel = observation[:3]  # Angular velocities
        ang_pos = observation[3:7]  # Angular position (quaternion)
        lin_vel = observation[7:10]  # Linear velocities
        lin_pos = observation[10:13]  # Linear positions
        prev_action = observation[13:17]  # Previous action
        aux_info = observation[17:]  # Auxiliary information

        # Extract specific altitude information
        altitude_error = observation[-2]
        current_altitude = observation[-1]

        # Extract roll from quaternion
        # Assuming quaternion is in [w, x, y, z] format
        # Quaternion to Euler angle conversion
        x, y, z, w = ang_pos
        # Roll (rotation around x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Convert to degrees for more intuitive control
        roll_deg = np.degrees(roll)
        print(roll_deg)

        # Return key state variables for control
        return np.array([
            roll_deg,       # Roll angle in degrees
            current_altitude,
            0,              # Yaw (placeholder, extract if needed)
            0  # Current altitude
        ])

    def predict(self, observation, deterministic=True):
        """
        Compute PID control signal based on the current observation

        Args:
            observation (dict or np.ndarray): Current state observation
            deterministic (bool): Ignored for PID controller, kept for API compatibility

        Returns:
            tuple: (action, state) where state is None for this simple controller
        """
        # Extract current state
        current_state = self._extract_state(self._format_state(observation))
        print(self.current_stage)
        print(observation)
        print(current_state)
        # Define target states
        target_roll = 0  # Level roll (0 degrees)
        target_altitude = 30  # Desired altitude (adjust as needed)

        # Initialize target state
        target_state = np.array([target_roll, target_altitude, 0, 0])

        # Compute error
        error = target_state - current_state

        # Separate control strategies based on current stage
        if self.current_stage == "LEVEL_ROLL":
            # Focus on leveling roll first
            error[1:] = 0  # Zero out other errors

            # Check if roll is sufficiently leveled
            if np.abs(current_state[0]) < 1:  # Within 0.1 degree of level
                self.current_stage = "ALTITUDE_CONTROL"

        elif self.current_stage == "ALTITUDE_CONTROL":
            error[1] = -1 * error[1]
            # Focus on altitude and maintaining level
            # error[0] = 0  # Keep roll at zero
            # error[1:3] = 0  # Maintain zero pitch and yaw

        # Proportional term
        P = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        I = self.ki * self.integral

        # Derivative term
        D = self.kd * (error - self.prev_error)

        # Compute control signal
        action = P + I + D

        # Clip the action to valid range
        action = np.clip(action, self.output_min, self.output_max)
        # apply max elevator
        max_elevator = 0.1
        action[1] = np.clip(action[1], -1 * max_elevator, max_elevator)

        # Always apply a constant thrust of 0.8
        action[-1] = 0.8

        # Update previous error for next iteration
        self.prev_error = error

        return action, None
def create_pid_controller(env, kp=1.0, ki=0.1, kd=0.01):
    """
    Factory function to create a PID controller

    Args:
        env (gym.Env): The gymnasium environment
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain

    Returns:
        PIDController: Initialized PID controller
    """
    return PIDController(env, kp, ki, kd)
