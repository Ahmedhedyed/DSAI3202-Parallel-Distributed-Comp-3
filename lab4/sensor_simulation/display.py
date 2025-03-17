# import os
# import time
# from temperature_simulation import temperature_queue, lock, condition, latest_temperature
# from temperature_processing import temperature_averages

# city = 'Doha'

# def initialize_display():
#     """
#     Initializes the display layout for showing current temperatures.
#     """
#     os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen
#     print("Current temperature for city:")
#     print(f"Latest Temperature: --°C")
#     print(f"{city} Average: --°C")

# def update_display():
#     """
#     Updates the display with the latest temperature and the average in real time.
#     """
#     last_displayed_temp = None  # Store last displayed temperature to avoid unnecessary screen refreshes

#     while True:
#         with lock:
#             # Wait for the latest temperature update
#             condition.wait_for(lambda: latest_temperature is not None)

#             # Avoid unnecessary refreshes
#             if latest_temperature == last_displayed_temp:
#                 continue

#             # Clear screen only if there's a new temperature update
#             os.system('cls' if os.name == 'nt' else 'clear')

#             print("Current temperature for city:")
#             print(f"Latest Temperature: {latest_temperature}°C")
            
#             # Display the latest computed average
#             avg_temp = temperature_averages[-1] if temperature_averages else '--'
#             print(f"{city} Average: {avg_temp}°C")

#             # Store the last displayed temperature to prevent redundant refreshes
#             last_displayed_temp = latest_temperature

#         time.sleep(1)  # Refresh every second


import os
import time

from globals import lock, condition, latest_temperature, temperature_averages, city


city = 'Doha'

def initialize_display():
    """
    Initializes the display layout for showing current temperatures.
    """
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console screen
    print("Current temperature for city:")
    print(f"Latest Temperature: --°C")
    print(f"{city} Average: --°C")

def update_display():
    """
    Updates the display with the latest temperature and the average in real time.
    """
    last_displayed_temp = None
    last_displayed_avg = None

    while True:
        with lock:
            # Wait until the latest temperature is not None
            condition.wait_for(lambda: latest_temperature is not None)

            current_temp = latest_temperature
            current_avg = temperature_averages[-1] if temperature_averages else '--'

            # Only refresh if there's a change in temperature OR average
            if current_temp != last_displayed_temp or current_avg != last_displayed_avg:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Current temperature for city:")
                print(f"Latest Temperature: {current_temp}°C")
                print(f"{city} Average: {current_avg}°C")

                # Update our "last displayed" values
                last_displayed_temp = current_temp
                last_displayed_avg = current_avg

        time.sleep(1)  # Refresh every second
