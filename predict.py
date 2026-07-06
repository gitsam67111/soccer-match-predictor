from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from SRC.teamStats import get_latest_team_stats
except ImportError:  # pragma: no cover - fallback for direct execution
    from teamStats import get_latest_team_stats

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH)


def predict_match_probabilities(home_team: str, away_team: str) -> dict[str, float | str]:
    """Return match outcome probabilities for a home/away matchup in percentage form."""
    home = get_latest_team_stats(home_team)
    away = get_latest_team_stats(away_team)

    game = pd.DataFrame(
        [
            {
                "home_form_points": home["form"],
                "away_form_points": away["form"],
                "home_elo": home["elo"],
                "away_elo": away["elo"],
            }
        ]
    )

    probabilities = model.predict_proba(game)[0]
    labels = model.classes_
    probability_by_label = {
        str(label): float(probability * 100) for label, probability in zip(labels, probabilities)
    }

    home_win_pct = (
        probability_by_label.get("H")
        or probability_by_label.get("home")
        or probability_by_label.get("Home")
        or 0.0
    )
    draw_pct = (
        probability_by_label.get("D")
        or probability_by_label.get("draw")
        or probability_by_label.get("Draw")
        or 0.0
    )
    away_win_pct = (
        probability_by_label.get("A")
        or probability_by_label.get("away")
        or probability_by_label.get("Away")
        or 0.0
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct": home_win_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_win_pct,
        "probabilities": probability_by_label,
    }


def format_prediction_result(result: dict[str, Any]) -> str:
    """Return a readable text block for display in the terminal or GUI."""
    return (
        f"{result['home_team']} win: {result['home_win_pct']:.1f}%\n"
        f"Draw: {result['draw_pct']:.1f}%\n"
        f"{result['away_team']} win: {result['away_win_pct']:.1f}%"
    )


def run_interface() -> None:
    """Launch a simple terminal interface for entering two teams."""
    print("Soccer Match Predictor")
    print("----------------------")
    home_team = input("Enter home team: ").strip()
    away_team = input("Enter away team: ").strip()

    result = predict_match_probabilities(home_team, away_team)
    print(f"\n{format_prediction_result(result)}")


def launch_gui() -> None:
    """Launch a polished desktop app using Tkinter."""
    import tkinter as tk
    from tkinter import messagebox

    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "E0.csv"
    team_frame = pd.read_csv(data_path)
    teams = sorted(set(team_frame["HomeTeam"]).union(set(team_frame["AwayTeam"])))

    window = tk.Tk()
    window.title("Soccer Match Predictor")
    window.geometry("420x260")
    window.resizable(False, False)

    title = tk.Label(window, text="Soccer Match Predictor", font=("Segoe UI", 16, "bold"))
    title.pack(pady=(16, 8))

    subtitle = tk.Label(window, text="Choose two teams to estimate the match outcome", font=("Segoe UI", 10))
    subtitle.pack(pady=(0, 12))

    home_var = tk.StringVar(window)
    away_var = tk.StringVar(window)
    home_var.set(teams[0])
    away_var.set(teams[1] if len(teams) > 1 else teams[0])

    selector_frame = tk.Frame(window)
    selector_frame.pack(fill="x", padx=24, pady=(0, 8))

    home_frame = tk.Frame(selector_frame)
    home_frame.pack(side="left", fill="x", expand=True, padx=(0, 8))
    tk.Label(home_frame, text="Home team", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    home_dropdown = tk.OptionMenu(home_frame, home_var, *teams)
    home_dropdown.config(width=18)
    home_dropdown.pack(anchor="w", pady=(4, 0))

    away_frame = tk.Frame(selector_frame)
    away_frame.pack(side="left", fill="x", expand=True)
    tk.Label(away_frame, text="Away team", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    away_dropdown = tk.OptionMenu(away_frame, away_var, *teams)
    away_dropdown.config(width=18)
    away_dropdown.pack(anchor="w", pady=(4, 0))

    def on_predict() -> None:
        home_team = home_var.get().strip()
        away_team = away_var.get().strip()
        if not home_team or not away_team:
            messagebox.showwarning("Input required", "Please choose both teams.")
            return

        try:
            result = predict_match_probabilities(home_team, away_team)
            messagebox.showinfo("Prediction", format_prediction_result(result))
        except Exception as exc:  # pragma: no cover - UI path
            messagebox.showerror("Prediction failed", str(exc))

    predict_button = tk.Button(
        window,
        text="Predict",
        command=on_predict,
        width=18,
        height=2,
        font=("Segoe UI", 11, "bold"),
    )
    predict_button.pack(pady=16)

    window.bind("<Return>", lambda event: on_predict())
    window.mainloop()


if __name__ == "__main__":
    run_interface()