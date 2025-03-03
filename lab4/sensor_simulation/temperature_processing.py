# temperature_processing.py

import threading
import time
from temperature_simulation import temperature_queue, lock
temperature_averages=[]

#
# Function to process temperatures and calculate averages
def process_temperatures():
    """
    Processes temperatures in the queue and calculates the average temperature.
    Updates the temperature_averages list every 5 seconds.
    """
    while True:
        time.sleep(5)  # Process every 5 seconds
        with lock:
            # Update the average temperature
            if not temperature_queue.empty():
                temperature_averages.append(temperature_queue.get())
            if len(temperature_averages) > 0:
                avg_temp = sum(temperature_averages) / len(temperature_averages)
            else:
                avg_temp = '--'
            temperature_averages[:] = [avg_temp]  # Update the average in place
            print(f"[PROCESS] Average Temperature Updated: {avg_temp}")  # Debugging print