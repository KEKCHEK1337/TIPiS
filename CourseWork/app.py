import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

MODEL_PATH = "nba_model.pkl"
STATS_PATH = "team_stats.json"


@st.cache_resource
def load_model_and_stats():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Модель не найдена: `{MODEL_PATH}`. Сначала выполните `python train.py`")
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            team_stats = json.load(f)
        return model, team_stats
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None


# =============== UI ===============
st.set_page_config(page_title="NBA Win Predictor", layout="centered")
st.title("Прогноз победы в NBA")
st.markdown("Выберите домашнюю и гостевую команду для предсказания исхода матча.")

model, team_stats = load_model_and_stats()

if model is None or team_stats is None:
    st.stop()

NBA_TEAMS = sorted(team_stats.keys())
if not NBA_TEAMS:
    st.error("Не найдено команд в данных.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Домашняя команда", NBA_TEAMS, index=NBA_TEAMS.index("LAL") if "LAL" in NBA_TEAMS else 0)
with col2:
    away_team = st.selectbox("Гостевая команда", NBA_TEAMS, index=NBA_TEAMS.index("GSW") if "GSW" in NBA_TEAMS else 0)

if st.button("Предсказать исход"):
    if home_team == away_team:
        st.error("Команды должны быть разными!")
    else:
        h = team_stats[home_team]
        a = team_stats[away_team]

        # Вектор признаков (10 значений)
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
            st.error(f"Ошибка предсказания: {e}")
            st.stop()

        # Вывод
        st.subheader("Вероятности победы")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(f"**{home_team}**", f"{home_win_prob:.1%}")
        with col_res2:
            st.metric(f"**{away_team}**", f"{1 - home_win_prob:.1%}")

        st.progress(float(home_win_prob))

        if home_win_prob > 0.55:
            st.success(f"Прогноз: **{home_team}** выиграет!")
        elif home_win_prob < 0.45:
            st.info(f"Прогноз: **{away_team}** выиграет!")
        else:
            st.warning("Прогноз: матч будет очень равным!")

        with st.expander("Средние показатели команд"):
            df_disp = pd.DataFrame({
                "Показатель": ["FG%", "FT%", "3PT%", "AST", "REB"],
                home_team: [
                    f"{h['FG_PCT']:.3f}",
                    f"{h['FT_PCT']:.3f}",
                    f"{h['FG3_PCT']:.3f}",
                    f"{h['AST']:.1f}",
                    f"{h['REB']:.1f}",
                ],
                away_team: [
                    f"{a['FG_PCT']:.3f}",
                    f"{a['FT_PCT']:.3f}",
                    f"{a['FG3_PCT']:.3f}",
                    f"{a['AST']:.1f}",
                    f"{a['REB']:.1f}",
                ],
            })
            st.table(df_disp)

st.markdown("---")
st.caption("Модель: XGBoost | Признаки: FG%, FT%, 3PT%, AST, REB")