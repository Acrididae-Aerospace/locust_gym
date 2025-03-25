import pprint
import sys
import time
from typing import Dict, Any
import colorama
from colorama import Fore, Back, Style


class DynamicTerminalTracker:
    def __init__(self):
        # Initialize colorama for cross-platform color support
        colorama.init(autoreset=True)

        # Store previous state to compare changes
        self.previous_state = {}

    def _get_value_color(self, current_value, prev_value):
        """Determine color based on value change."""
        if isinstance(current_value, (int, float)):
            if prev_value is None:
                return Back.WHITE  # Initial display
            elif current_value > prev_value:
                return Back.GREEN  # Increased
            elif current_value < prev_value:
                return Back.RED  # Decreased

        if isinstance(current_value, bool):
            if prev_value is None:
                return ""
            elif current_value and not prev_value:
                return Back.GREEN  # Turned True
            elif not current_value and prev_value:
                return Back.RED  # Turned False

        return ""

    def display(self, data: Dict[str, Any], total: float = None):
        """
        Display data with dynamic coloring based on changes.

        Args:
            data (dict): Dictionary of values to display
            total (float, optional): Total value to display at the top
        """
        # Clear the screen
        print("\033[H\033[J", end="")

        # Display total if provided
        if total is not None:
            total_color = self._get_value_color(total, self.previous_state.get("total"))
            print(f"{total_color}Total: {total}{Style.RESET_ALL}")

        # Iterate through data and display with dynamic coloring
        for key, value in sorted(data.items()):
            # Get color based on value change
            color = self._get_value_color(value, self.previous_state.get(key))

            # Format different types of values
            if isinstance(value, float):
                formatted_value = f"{value:.4e}"
            elif isinstance(value, bool):
                formatted_value = str(value)
            else:
                formatted_value = str(value)

            # Print with color
            print(f"{color}{key}: {formatted_value}{Style.RESET_ALL}")

        # Update previous state
        self.previous_state = {**data, "total": total}

    def track(self, data: Dict[str, Any], total: float = None, interval: float = 0.5):
        """
        Continuously track and display data changes.

        Args:
            data (dict): Dictionary of values to track
            total (float, optional): Total value to display
            interval (float): Time between updates in seconds
        """
        try:
            while True:
                self.display(data, total)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nTracking stopped.")
