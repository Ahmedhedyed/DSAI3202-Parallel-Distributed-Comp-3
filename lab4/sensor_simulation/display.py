
import os
from temperature_simulation import temperature_queue, lock, condition, latest_temperature
import time
from temperature_processing import temperature_averages

city = 'Doha'

# Function to initialize the display layout
def initialize_display():
    """
    Initializes the display layout for showing current temperatures.
    """
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen
    print("Current temperature for city:")
    print(f"Latest Temperature: --°C")
    print(f"{city} Average: --°C")

# Function to update the display with the latest temperature and average
def update_display():
    """
    Updates the display with the latest temperature and the average in place.
    """
    while True:
        with lock:
            # Wait until the latest temperature is updated
            condition.wait_for(lambda: latest_temperature is not None)
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen
            print("Current temperature for city:")
            # Display the latest temperature and average
            print(f"Latest Temperature: {latest_temperature}°C")
            avg_temp = temperature_averages[0] if len(temperature_averages) > 0 else '--'
            print(f"{city} Average: {avg_temp}°C")
        time.sleep(1)  # Refresh the display every 1 second
