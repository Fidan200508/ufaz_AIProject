import json
from plot_utils import plot_training_curves, plot_optimizer_summary
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "results" / "digits_optimizer_study.json"
    save_dir = script_dir.parent / "figures"

    with open(data_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # --- ADD THIS CHECK ---
    # If results is a single dictionary, wrap it in a list
    if isinstance(results, dict):
        results = [results]
    # ----------------------

    plot_training_curves(results, save_dir=save_dir)
    plot_optimizer_summary(results, save_dir=save_dir)

    print(f"Plots saved in {save_dir}")

if __name__ == "__main__":
    main()