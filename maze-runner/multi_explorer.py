from multiprocessing import Pool
from src.maze import create_maze
from src.explorer import Explorer
import time
from time import perf_counter




def run_single_explorer(explorer_id):
    """
    Runs one maze explorer and returns stats.
    """
    maze = create_maze(50, 50, "static")  # or "random"
    explorer = Explorer(maze, visualize=False)

    # start = time.time()
    start = perf_counter()
    time_taken, moves = explorer.solve()
    # end = time.time()
    end = time.perf_counter()

    stats = {
        "id": explorer_id,
        "time_taken": time_taken,
        "moves": len(moves),
        "backtracks": explorer.backtrack_count,
        "runtime": round(end - start, 2),
    }

    return stats
