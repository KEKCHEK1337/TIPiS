# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "nba_model.pkl"
STATS_PATH = "team_stats.json"
GAMES_PATH = "games.csv"
TEAMS_PATH = "teams.csv"


def load_teams_dict():
    if not os.path.exists(TEAMS_PATH):
        st.error(f"Файл `{TEAMS_PATH}` не найден.")
        return None

    try:
        teams_df = pd.read_csv(TEAMS_PATH)
        if "TEAM_ID" not in teams_df.columns:
            st.error("В `teams.csv` нет колонки `TEAM_ID`")
            return None

        abbr_col = None
        for col in ["ABBREVIATION", "TEAM_ABBREVIATION"]:
            if col in teams_df.columns:
                abbr_col = col
                break
        if not abbr_col:
            st.error("В `teams.csv` нет колонки с аббревиатурой")
            return None

        teams_df = teams_df[["TEAM_ID", abbr_col]].drop_duplicates()
        teams_df["TEAM_ID"] = teams_df["TEAM_ID"].astype(int)
        team_id_to_abbr = dict(zip(teams_df["TEAM_ID"], teams_df[abbr_col]))
        return team_id_to_abbr

    except Exception as e:
        st.error(f"Ошибка чтения `teams.csv`: {e}")
        return None


def train_and_save_model():
    st.info("Обучаем модель…")

    team_id_to_abbr = load_teams_dict()
    if team_id_to_abbr is None:
        return None, None

    if not os.path.exists(GAMES_PATH):
        st.error(f"Файл `{GAMES_PATH}` не найден.")
        return None, None

    try:
        df = pd.read_csv(GAMES_PATH)
        st.write(f"Загружено {len(df)} матчей")
    except Exception as e:
        st.error(f"Ошибка чтения `{GAMES_PATH}`: {e}")
        return None, None

    df.columns = df.columns.str.lower()

    required_cols = {
        "team_id_home", "team_id_away",
        "pts_home", "pts_away"
    }
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Отсутствуют колонки: {missing}")
        st.write("Доступные колонки:", list(df.columns))
        return None, None

    df["team_abbr_home"] = df["team_id_home"].map(team_id_to_abbr)
    df["team_abbr_away"] = df["team_id_away"].map(team_id_to_abbr)

    before = len(df)
    df = df.dropna(subset=["team_abbr_home", "team_abbr_away"])
    after = len(df)
    if before > after:
        st.warning(f"Удалено {before - after} матчей с неизвестными TEAM_ID")

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
        st.error(f"Отсутствуют признаки: {missing_features}")
        st.write("Доступные колонки:", list(df.columns))
        return None, None

    df = df.dropna(subset=feature_cols + ["home_team_wins"])

    X = df[feature_cols]
    y = df["home_team_wins"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Модель обучена! Точность: {acc:.2%}")

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

    return model, team_stats


@st.cache_resource
def load_model_and_stats():
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                team_stats = json.load(f)
            return model, team_stats
        except Exception as e:
            st.warning(f"Ошибка загрузки: {e}. Переобучаем…")

    return train_and_save_model()


# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="NBA Win Predictor", layout="centered")
st.title("NBA Win Probability Predictor")
st.markdown("Выберите команды для предсказания победы.")

model, team_stats = load_model_and_stats()

if model is None or team_stats is None:
    st.error("Не удалось инициализировать модель.")
    st.stop()

NBA_TEAMS = sorted(team_stats.keys())
if not NBA_TEAMS:
    st.error("Не найдено команд.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Домашняя команда", NBA_TEAMS, index=NBA_TEAMS.index("LAL") if "LAL" in NBA_TEAMS else 0)
with col2:
    away_team = st.selectbox("Гостевая команда", NBA_TEAMS, index=NBA_TEAMS.index("GSW") if "GSW" in NBA_TEAMS else 0)

if st.button("Предсказать"):
    if home_team == away_team:
        st.error("Команды должны быть разными!")
    else:
        h = team_stats[home_team]
        a = team_stats[away_team]

        features = np.array([[
            h["FG_PCT"], a["FG_PCT"],
            h["FT_PCT"], a["FT_PCT"],
            h["FG3_PCT"], a["FG3_PCT"],
            h["AST"], a["AST"],
            h["REB"], a["REB"],
        ]])

        try:
            prob = model.predict_proba(features)[0]
            home_win_prob = prob[1]
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

        st.subheader("Вероятности")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"**{home_team}**", f"{home_win_prob:.1%}")
        with col2:
            st.metric(f"**{away_team}**", f"{1 - home_win_prob:.1%}")

        st.progress(float(home_win_prob))

        if home_win_prob > 0.55:
            st.success(f"Прогноз: **{home_team}** выиграет!")
        elif home_win_prob < 0.45:
            st.info(f"Прогноз: **{away_team}** выиграет!")
        else:
            st.warning("Очень равный матч!")

        with st.expander("Показатели команд"):
            df_disp = pd.DataFrame({
                "Показатель": ["FG%", "FT%", "3PT%", "AST", "REB"],
                home_team: [f"{h[k]:.3f}" if k != "AST" and k != "REB" else f"{h[k]:.1f}" for k in ["FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]],
                away_team: [f"{a[k]:.3f}" if k != "AST" and k != "REB" else f"{a[k]:.1f}" for k in ["FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]],
            })
            st.table(df_disp)

st.markdown("---")
st.caption("Используются: FG%, FT%, 3PT%, AST, REB")