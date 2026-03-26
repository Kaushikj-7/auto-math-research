import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from scripts.init_db import setup_db
from scripts.status import main as status_main
from optimizers.mutator import run_mutator
from training.worker import run_worker
from configs.research_config import ResearchConfig


def main():
    parser = argparse.ArgumentParser(description="Auto Math Research Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init
    subparsers.add_parser("init", help="Initialize the database")

    # Status
    subparsers.add_parser("status", help="Show current status")

    # Worker
    worker_parser = subparsers.add_parser("worker", help="Run a worker")
    worker_parser.add_argument(
        "--device", default="cuda", help="Device to use (cuda/cpu)"
    )

    # Mutator
    mutator_parser = subparsers.add_parser("mutator", help="Run the mutator")

    args = parser.parse_args()

    if args.command == "init":
        print(f"Initializing database at {ResearchConfig.DB_PATH}")
        setup_db()
        print("Done.")
    elif args.command == "status":
        status_main()
    elif args.command == "worker":
        run_worker(device=args.device)
    elif args.command == "mutator":
        run_mutator()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
