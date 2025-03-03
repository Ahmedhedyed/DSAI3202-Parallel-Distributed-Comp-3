# temperature_simulation.py

import random
import time
import threading
import queue

# Global variables
latest_temperature = None
temperature_queue = queue.Queue()
lock = threading.Lock()
condition = threading.Condition(lock)

# Function to simulate temperature readings for a city
def simulate_city_temperature():
    """
    Simulates temperature readings for the city.
    Updates the latest_temperature variable every second.
    """
    global latest_temperature
    while True:
        temperature = random.randint(15, 40)  # Random temperature between 15 and 40
        with lock:
            latest_temperature = temperature
            temperature_queue.put(temperature)  # Add the temperature to the queue
            print(f"[SIMULATION] Latest Temperature Updated: {latest_temperature}")  # Debugging print
            condition.notify_all()  # Notify the display thread that the temperature was updated
        time.sleep(1)  # Simulate a delay of 1 second for each city reading