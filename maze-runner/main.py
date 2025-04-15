"""
Main entry point for the maze runner game.
"""

import argparse
from src.game import run_game
from src.explorer import Explorer, AStarExplorer, MemoryExplorer
from multiprocessing import Pool
from src.explorer import Explorer
import multi_explorer
from src.maze import create_maze



def run_parallel_explorers(num_explorers=4):
    print(f"Running {num_explorers} maze explorers in parallel...\n")
    
    with Pool(processes=num_explorers) as pool:
        results = pool.map(multi_explorer.run_single_explorer, range(num_explorers))

    # Print summary
    print("=== Explorer Results ===")
    for res in results:
        print(f"Explorer #{res['id']} → Time: {res['time_taken']:.2f}s | "
              f"Moves: {res['moves']} | Backtracks: {res['backtracks']} | "
              f"Runtime: {res['runtime']}s")

    # Determine best performer (shortest path)
    best = min(results, key=lambda x: x["moves"])
    print("\nBest Explorer:")
    print(f"Explorer #{best['id']} with {best['moves']} moves "
          f"in {best['time_taken']:.2f} seconds.")



def main():
    parser = argparse.ArgumentParser(description="Maze Runner Game")
    parser.add_argument("--type", choices=["random", "static"], default="random",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--auto", action="store_true",
                        help="Run automated maze exploration")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the automated exploration in real-time")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Run multiple explorers in parallel (set number)")
    #Added
    parser.add_argument("--solver", choices=["right", "astar", "memory"], default="right",
                        help="Choose the explorer strategy")


    
    args = parser.parse_args()
    
    if args.parallel > 0:
        run_parallel_explorers(num_explorers=args.parallel)
        return


    if args.auto:
        # Create maze and run automated exploration
        maze = create_maze(args.width, args.height, args.type)

 # Select solver Added to chose between algorethem
        if args.solver == "right":
            explorer = Explorer(maze, visualize=args.visualize)
        elif args.solver == "astar":
            explorer = AStarExplorer(maze, visualize=args.visualize)
        elif args.solver == "memory":
            explorer = MemoryExplorer(maze, visualize=args.visualize)
        else:
            raise ValueError("Invalid solver")
        
        time_taken, moves = explorer.solve()
        print(f"Maze solved in {time_taken:.2f} seconds")
        print(f"Number of moves: {len(moves)}")
        if args.type == "static":
            print("Note: Width and height arguments were ignored for the static maze")
    else:
        # Run the interactive game
        run_game(maze_type=args.type, width=args.width, height=args.height)



if __name__ == "__main__":
    main()