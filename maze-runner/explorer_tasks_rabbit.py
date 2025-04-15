from celery import Celery
from src.maze import create_maze
from src.explorer import Explorer
import time

# Configure Celery to use RabbitMQ
app = Celery(
    'explorer_tasks_rabbit',
    broker='pyamqp://guest@localhost//',   # RabbitMQ broker
    backend='rpc://'  # optional backend to receive results
)

@app.task
def run_explorer_task(explorer_id):
    maze = create_maze(50, 50, "random")
    explorer = Explorer(maze, visualize=False)

    start = time.perf_counter()
    time_taken, moves = explorer.solve()
    end = time.perf_counter()

    return {
        "id": explorer_id,
        "time_taken": time_taken,
        "moves": len(moves),
        "backtracks": explorer.backtrack_count,
        "runtime": round(end - start, 2)
    }
