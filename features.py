from collections import defaultdict, deque

import pandas as pd

def _get_match_points(row: pd.Series) -> tuple[int, int]:
    """Return the points earned by the home and away teams for a single match row."""
    if "FTR" in row.index:
        result = row["FTR"]
        if result == "H":
            return 3, 0
        if result == "D":
            return 1, 1
        if result == "A":
            return 0, 3

    home_goals = row.get("FTHG", row.get("home_score", row.get("HomeGoals")))
    away_goals = row.get("FTAG", row.get("away_score", row.get("AwayGoals")))

    if home_goals is None or away_goals is None:
        raise ValueError("Could not infer match result from the available columns.")

    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def add_previous_5_match_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling previous-5-match points features for home and away teams."""
    data = df.copy()

    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    team_form = defaultdict(lambda: deque(maxlen=5))
    home_prev_points = []
    away_prev_points = []

    for _, row in data.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_points, away_points = _get_match_points(row)

        home_prev_points.append(sum(team_form[home_team]))
        away_prev_points.append(sum(team_form[away_team]))

        team_form[home_team].append(home_points)
        team_form[away_team].append(away_points)

    data["home_form_points"] = home_prev_points
    data["away_form_points"] = away_prev_points
    return data


def add_elo_ratings(
    df: pd.DataFrame,
    start_rating: int = 1500,
    base_k: int = 20,
    gd_weight: int = 4,
) -> pd.DataFrame:
    """Add evolving ELO ratings for home and away teams based on each match result and goal difference."""
    data = df.copy()

    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    ratings = defaultdict(lambda: start_rating)
    home_elo = []
    away_elo = []
    home_elo_change = []
    away_elo_change = []

    for _, row in data.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_goals = row.get("FTHG", row.get("home_score", row.get("HomeGoals")))
        away_goals = row.get("FTAG", row.get("away_score", row.get("AwayGoals")))

        if home_goals is None or away_goals is None:
            raise ValueError("Could not infer match goals from the available columns.")

        home_rating_before = ratings[home_team]
        away_rating_before = ratings[away_team]

        home_expected = 1 / (1 + 10 ** ((away_rating_before - home_rating_before) / 400))
        away_expected = 1 - home_expected

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

        home_elo.append(home_rating_before)
        away_elo.append(away_rating_before)
        home_elo_change.append(home_change)
        away_elo_change.append(away_change)

    data["home_elo"] = home_elo
    data["away_elo"] = away_elo
    data["home_elo_change"] = home_elo_change
    data["away_elo_change"] = away_elo_change
    return data


if __name__ == "__main__":
    df = pd.read_csv("data/raw/E0.csv")
    df = add_previous_5_match_points(df)
    df = add_elo_ratings(df)

    print(
        df[
            [
                "HomeTeam",
                "AwayTeam",
                "home_form_points",
                "away_form_points",
                "home_elo",
                "away_elo",
            ]
        ].head(100)
    )
    