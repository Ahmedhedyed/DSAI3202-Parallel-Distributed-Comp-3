# globals.py

import queue
import threading

# Shared data
latest_temperature = None
temperature_averages = []
temperature_queue = queue.Queue()

# Synchronization tools
lock = threading.Lock()
condition = threading.Condition(lock)

# City name (shared across modules)
city = "Doha"
