import pandas as pd
import numpy as np
import joblib
import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

GAMES_PATH = "games.csv"
TEAMS_PATH = "teams.csv"
MODEL_PATH = "nba_model.pkl"
STATS_PATH = "team_stats.json"


def load_teams_dict():
    if not os.path.exists(TEAMS_PATH):
        raise FileNotFoundError(f"Файл `{TEAMS_PATH}` не найден. Скачайте его с Kaggle.")

    teams_df = pd.read_csv(TEAMS_PATH)
    if "TEAM_ID" not in teams_df.columns:
        raise ValueError("В `teams.csv` нет колонки `TEAM_ID`")

    abbr_col = None
    for col in ["ABBREVIATION", "TEAM_ABBREVIATION"]:
        if col in teams_df.columns:
            abbr_col = col
            break
    if not abbr_col:
        raise ValueError("В `teams.csv` нет колонки с аббревиатурой (ожидались: ABBREVIATION или TEAM_ABBREVIATION)")

    teams_df = teams_df[["TEAM_ID", abbr_col]].drop_duplicates()
    teams_df["TEAM_ID"] = teams_df["TEAM_ID"].astype(int)
    return dict(zip(teams_df["TEAM_ID"], teams_df[abbr_col]))


def main():
    print("Загружаем данные")
    team_id_to_abbr = load_teams_dict()
    print(f"Загружено {len(team_id_to_abbr)} команд")

    if not os.path.exists(GAMES_PATH):
        raise FileNotFoundError(f"Файл `{GAMES_PATH}` не найден.")

    df = pd.read_csv(GAMES_PATH)
    print(f"Загружено {len(df)} матчей")
    df.columns = df.columns.str.lower()

    required_id_cols = {"team_id_home", "team_id_away"}
    missing = required_id_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}. Доступные: {list(df.columns)}")

    df["team_abbr_home"] = df["team_id_home"].map(team_id_to_abbr)
    df["team_abbr_away"] = df["team_id_away"].map(team_id_to_abbr)
    df = df.dropna(subset=["team_abbr_home", "team_abbr_away"])
    print(f"После фильтрации: {len(df)} матчей")

    df["home_team_wins"] = (df["pts_home"] > df["pts_away"]).astype(int)

    feature_cols = [
        "fg_pct_home", "fg_pct_away",
        "ft_pct_home", "ft_pct_away",
        "fg3_pct_home", "fg3_pct_away",
        "ast_home", "ast_away",
        "reb_home", "reb_away",
    ]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Отсутствуют признаки: {missing_features}")

    df = df.dropna(subset=feature_cols + ["home_team_wins"])
    X = df[feature_cols]
    y = df["home_team_wins"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Обучаем модель XGBoost")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nРезультаты на тесте:")
    print(f"   Accuracy: {acc:.3f}  ({acc:.1%})")
    print(f"   ROC-AUC : {auc:.3f}")
    print("\nПодробный отчёт:")
    print(classification_report(y_test, y_pred))

    print("Собираем среднюю статистику по командам")
    home_agg = df.groupby("team_abbr_home")[
        ["fg_pct_home", "ft_pct_home", "fg3_pct_home", "ast_home", "reb_home"]
    ].mean()
    away_agg = df.groupby("team_abbr_away")[
        ["fg_pct_away", "ft_pct_away", "fg3_pct_away", "ast_away", "reb_away"]
    ].mean()

    home_agg.columns = ["FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]
    away_agg.columns = ["FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]

    team_stats = {}
    all_teams = set(home_agg.index) | set(away_agg.index)
    for team in all_teams:
        h = home_agg.loc[team] if team in home_agg.index else pd.Series([0]*5, index=home_agg.columns)
        a = away_agg.loc[team] if team in away_agg.index else pd.Series([0]*5, index=away_agg.columns)
        avg = (h + a) / 2
        team_stats[team] = avg.to_dict()

    joblib.dump(model, MODEL_PATH)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=2)

    print(f"\nМодель сохранена: {MODEL_PATH}")
    print(f"Статистика команд: {STATS_PATH}")
    print("\nТеперь запустите: streamlit run app.py")


if __name__ == "__main__":
    main()