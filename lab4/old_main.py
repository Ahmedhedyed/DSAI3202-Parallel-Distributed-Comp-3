import random
import time
import threading
import os
import queue

# Global variables for shared data
latest_temperature = None
temperature_averages = []
temperature_queue = queue.Queue()

# Synchronization tools
lock = threading.Lock()  # Simple lock for synchronizing access to shared resources
condition = threading.Condition(lock)

# City name
city = 'Doha'  # The city for temperature simulation

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
            if temperature_queue.qsize() > 0:
                temperature_averages.append(temperature_queue.get())
            if len(temperature_averages) > 0:
                avg_temp = sum(temperature_averages) / len(temperature_averages)
            else:
                avg_temp = '--'
            temperature_averages[:] = [avg_temp]  # Update the average in place
            print(f"[PROCESS] Average Temperature Updated: {avg_temp}")  # Debugging print

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

# Main program to start all threads and simulation
def main():
    # Initialize display
    initialize_display()
    
    # Create and start the city temperature simulation thread
    city_thread = threading.Thread(target=simulate_city_temperature)
    city_thread.daemon = True  # Daemonize the thread so it terminates when the main program ends
    city_thread.start()

    # Create and start the processing thread
    process_thread = threading.Thread(target=process_temperatures)
    process_thread.daemon = True
    process_thread.start()

    # Create and start the display update thread
    display_thread = threading.Thread(target=update_display)
    display_thread.daemon = True
    display_thread.start()

    # Keep the main thread running to allow daemon threads to continue
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
