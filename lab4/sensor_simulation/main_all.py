# main_all.py

# Main program to start all threads and simulation
import time
import threading
from display import initialize_display, update_display
from temperature_simulation import simulate_city_temperature
from temperature_processing import process_temperatures


# Main program to start all threads and simulation
def main():
    # Initialize display
    initialize_display()
    
    # Create and start the city temperature simulation thread
    city_thread = threading.Thread(target=simulate_city_temperature, daemon=True)
    city_thread.start()

    # Create and start the processing thread
    process_thread = threading.Thread(target=process_temperatures, daemon=True)
    process_thread.start()

    # Create and start the display update thread
    display_thread = threading.Thread(target=update_display, daemon=True)
    display_thread.start()

    # Keep the main thread running to allow daemon threads to continue
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
