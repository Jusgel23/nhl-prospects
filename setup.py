"""Quick-start setup script — installs dependencies and initializes the DB."""
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    print("=== NHL Prospect System Setup ===\n")

    print("[1/3] Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

    print("\n[2/3] Creating data directories...")
    for d in ["data/raw", "data/processed", "data/historical",
              "data/models", "data/reports", "notebooks"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")

    print("\n[3/3] Initializing database...")
    from src.data.database import init_db
    init_db()
    print("  ✓ data/prospects.db created")

    print("\n=== Setup complete ===")
    print("\nNext steps:")
    print("  1. Download the Kaggle dataset to data/historical/nhl_draft_1963_2022.csv")
    print("     → https://www.kaggle.com/datasets/mattop/nhl-draft-hockey-player-data-1963-2022")
    print("  2. Run:  python main.py collect")
    print("  3. Run:  python main.py process")
    print("  4. Run:  python main.py train")
    print("  5. Run:  python main.py rank --top 50")
    print("  6. Run:  python main.py report --player 'Macklin Celebrini'")
    print("\n  Or run the full pipeline in one shot:")
    print("       python main.py pipeline")


if __name__ == "__main__":
    main()
