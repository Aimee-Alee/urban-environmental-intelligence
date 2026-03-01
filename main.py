"""
main.py — Single entry-point orchestrator for the Urban Environmental Intelligence pipeline.
Runs all four analytical tasks in sequence after data has been ingested.

Usage:
    python main.py            # Run all tasks
    python main.py --task 1   # Run a single task (1-4)
    python main.py --ingest   # Run data ingestion only
"""

import sys
import argparse
import os

# Add src/ to path so tasks can import each other's modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Urban Environmental Intelligence — Analysis Pipeline"
    )
    parser.add_argument(
        "--task", type=int, choices=[1, 2, 3, 4],
        help="Run a single task (1-4). If omitted, all tasks are run."
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Run the data ingestion pipeline (data_loader.py) instead of analysis."
    )
    args = parser.parse_args()

    if args.ingest:
        print("=" * 60)
        print("PHASE 0: Data Ingestion (OpenAQ API)")
        print("=" * 60)
        import data_loader  # noqa — runs as __main__
        return

    # Check data exists
    if not os.path.exists("data/final_dataset.parquet"):
        print("ERROR: data/final_dataset.parquet not found.")
        print("Please run data ingestion first:")
        print("  python main.py --ingest")
        sys.exit(1)

    # ── Task Runner ────────────────────────────────────────────────────────────
    from task1_pca         import run_task1
    from task2_temporal    import run_task2
    from task3_distribution import run_task3
    from task4_integrity   import run_task4

    tasks = {
        1: run_task1,
        2: run_task2,
        3: run_task3,
        4: run_task4,
    }

    if args.task:
        print(f"\n{'='*60}")
        print(f"Running Task {args.task} only")
        print(f"{'='*60}\n")
        tasks[args.task]()
    else:
        print("\n" + "=" * 60)
        print("Running Full Analytical Pipeline (Tasks 1 → 4)")
        print("=" * 60 + "\n")
        for task_num, task_fn in tasks.items():
            print(f"\n{'─'*60}")
            task_fn()
        print("\n" + "=" * 60)
        print("All tasks complete. Results saved to results/")
        print("=" * 60)
        print("\nTo launch the interactive dashboard:")
        print("  streamlit run src/app.py")


if __name__ == "__main__":
    main()
