from mpi4py import MPI
from src.maze import create_maze
from src.explorer import Explorer
import time
import random
from time import perf_counter


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

maze = create_maze(50, 50, "static")
explorer = Explorer(maze, visualize=False)

# start = time.time()
start = perf_counter()
time_taken, moves = explorer.solve()
# end = time.time()
end = time.perf_counter()


result = {
    "rank": rank,
    "time_taken": time_taken,
    "moves": len(moves),
    "backtracks": explorer.backtrack_count,
    "runtime": round(end - start, 2)
}

results = comm.gather(result, root=0)

if rank == 0:
    print("\n=== MPI Explorer Results ===")
    for r in results:
        print(f"Explorer #{r['rank']} → Time: {r['time_taken']:.2f}s | "
              f"Moves: {r['moves']} | Backtracks: {r['backtracks']} | Runtime: {r['runtime']}s")

    best = min(results, key=lambda x: x["moves"])
    print(f"\n Best Explorer: #{best['rank']} with {best['moves']} moves.")
