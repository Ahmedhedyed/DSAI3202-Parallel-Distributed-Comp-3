from explorer_tasks_rabbit import run_explorer_task
import pandas as pd

def run_tasks(num=6):
    print(f"Dispatching {num} maze explorers via Celery + RabbitMQ...\n")
    tasks = [run_explorer_task.delay(i) for i in range(num)]

    print("Waiting for results...")
    results = [task.get(timeout=60) for task in tasks]

    # Print Results
    print(f"\n{'ID':<5} {'Time (s)':<10} {'Moves':<10} {'Backtracks':<12} {'Runtime (s)':<12}")
    for r in results:
        print(f"{r['id']:<5} {r['time_taken']:<10.2f} {r['moves']:<10} {r['backtracks']:<12} {r['runtime']:<12.2f}")

    # Find best
    best = min(results, key=lambda x: x["moves"])
    print(f"\n🏆 Best Explorer: #{best['id']} with {best['moves']} moves in {best['time_taken']:.2f} seconds.")

    # Optional: save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("rabbitmq_explorer_results.csv", index=False)

if __name__ == "__main__":
    run_tasks(6)
