from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .features import _get_match_points
except ImportError:  # pragma: no cover - fallback for direct execution
    from features import _get_match_points


def get_latest_team_stats(
    team: str | None = None,
    df: pd.DataFrame | None = None,
    start_rating: float = 1500.0,
    base_k: float = 20.0,
    gd_weight: float = 4.0,
) -> dict[str, float] | pd.DataFrame:
    """Return the latest ELO rating and form points for one team or for every team.

    The form value is the sum of the last 5 match points earned by that team.
    """
    if df is None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "E0.csv"
        df = pd.read_csv(data_path)

    data = df.copy()
    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    if {"HomeTeam", "AwayTeam"}.difference(data.columns):
        raise ValueError("The dataset must contain HomeTeam and AwayTeam columns.")

    ratings: dict[str, float] = defaultdict(lambda: start_rating)
    form_points: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
    latest_stats: dict[str, dict[str, float]] = {}

    for _, row in data.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_points, away_points = _get_match_points(row)

        home_rating_before = ratings[home_team]
        away_rating_before = ratings[away_team]

        home_expected = 1 / (1 + 10 ** ((away_rating_before - home_rating_before) / 400))
        away_expected = 1 - home_expected

        home_goals = row.get("FTHG", row.get("home_score", row.get("HomeGoals")))
        away_goals = row.get("FTAG", row.get("away_score", row.get("AwayGoals")))

        if home_goals is None or away_goals is None:
            raise ValueError("Could not infer match goals from the available columns.")

        goal_diff = int(home_goals) - int(away_goals)
        if goal_diff > 0:
            home_score, away_score = 1.0, 0.0
        elif goal_diff < 0:
            home_score, away_score = 0.0, 1.0
        else:
            home_score, away_score = 0.5, 0.5

        k_factor = base_k + gd_weight * abs(goal_diff)
        home_change = k_factor * (home_score - home_expected)
        away_change = k_factor * (away_score - away_expected)

        home_rating_after = home_rating_before + home_change
        away_rating_after = away_rating_before + away_change

        ratings[home_team] = home_rating_after
        ratings[away_team] = away_rating_after

        form_points[home_team].append(home_points)
        form_points[away_team].append(away_points)

        latest_stats[home_team] = {
            "elo": float(home_rating_after),
            "form": float(sum(form_points[home_team])),
        }
        latest_stats[away_team] = {
            "elo": float(away_rating_after),
            "form": float(sum(form_points[away_team])),
        }

    if team is not None:
        team_name = str(team).strip()
        if team_name not in latest_stats:
            raise KeyError(f"Team '{team_name}' was not found in the dataset.")
        return latest_stats[team_name]

    return pd.DataFrame.from_dict(latest_stats, orient="index").rename_axis("team")
