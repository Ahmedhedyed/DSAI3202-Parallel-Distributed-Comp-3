
from .globals import latest_temperature, temperature_queue, lock, condition
import random
import time

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
            temperature_queue.put(temperature)  # Add temperature to queue
            
            print(f"[SIMULATION] Latest Temperature: {latest_temperature}°C")
            condition.notify_all()  # Notify display thread

        time.sleep(1)  # Simulate a delay of 1 second per reading
