import threading
import time


from globals import temperature_queue, temperature_averages, lock


MAX_AVERAGE_SIZE = 10  # Store only the last 10 readings

def process_temperatures():
    """
    Processes all available temperatures in the queue and calculates a moving average every 5 seconds.
    """
    while True:
        time.sleep(5)  # Process temperatures every 5 seconds

        with lock:
            temps = []

            # Retrieve all values from the queue
            while not temperature_queue.empty():
                temps.append(temperature_queue.get())

            if temps:
                temperature_averages.extend(temps)

                # Keep only the last 10 readings for moving average calculation
                if len(temperature_averages) > MAX_AVERAGE_SIZE:
                    temperature_averages[:] = temperature_averages[-MAX_AVERAGE_SIZE:]

                # Calculate the moving average
                avg_temp = sum(temperature_averages) / len(temperature_averages)
            else:
                avg_temp = '--'  # No data available

            print(f"[PROCESS] Average Temperature Updated: {avg_temp}")  # Debugging print
