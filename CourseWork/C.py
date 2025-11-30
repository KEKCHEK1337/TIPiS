# app.py — Drug Recommender для датасета с Medicine Name, Uses, Excellent Review %
import streamlit as st
import pandas as pd

st.set_page_config(page_title="💊 Drug Recommender (Uses-based)", layout="centered")
st.title("💊 Рекомендация лекарств по симптому")
st.markdown("Введите симптом или заболевание (например: *Fever*, *Headache*, *Diabetes*) — и получите подходящие лекарства.")

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('drug_dataset.csv')
    except FileNotFoundError:
        st.error("❌ Файл 'drug_dataset.csv' не найден.")
        return None

    # Проверка: есть ли нужные столбцы?
    expected = ['Medicine Name', 'Uses', 'Excellent Review %']
    for col in expected:
        if col not in df.columns:
            st.error(f"Столбец '{col}' не найден. Найденные: {list(df.columns)}")
            return None

    # Очистка и преобразование
    df = df.dropna(subset=['Uses', 'Medicine Name'])
    df['Uses'] = df['Uses'].astype(str)
    df['Excellent Review %'] = pd.to_numeric(df['Excellent Review %'], errors='coerce').fillna(0)

    # Создаём список всех заболеваний
    all_conditions = set()
    df['Uses_list'] = df['Uses'].str.split(',')
    for uses in df['Uses_list']:
        for u in uses:
            all_conditions.add(u.strip().lower())

    return df, sorted(all_conditions)

# Загружаем данные
result = load_and_prepare_data()
if result is None:
    st.stop()

df, all_conditions = result

# Пользовательский ввод
user_input = st.selectbox(
    "Выберите или введите симптом/заболевание:",
    options=[""] + all_conditions,
    format_func=lambda x: x if x else "🔍 Начните ввод..."
)

if user_input:
    target = user_input.strip().lower()
    recommendations = []

    for _, row in df.iterrows():
        uses_list = [u.strip().lower() for u in row['Uses'].split(',')]
        if target in uses_list:
            recommendations.append({
                'Medicine Name': row['Medicine Name'],
                'Excellent %': row['Excellent Review %'],
                'Uses': row['Uses']
            })

    if recommendations:
        rec_df = pd.DataFrame(recommendations).sort_values(by='Excellent %', ascending=False).head(10)
        st.subheader(f"Лекарства при: **{user_input}**")
        for _, row in rec_df.iterrows():
            st.markdown(f"""
            **💊 {row['Medicine Name']}**  
            👍 Отличные отзывы: **{row['Excellent %']:.1f}%**  
            ℹ️ Применяется при: {row['Uses']}
            """)
            st.divider()
    else:
        st.warning("Не найдено лекарств для этого симптома. Попробуйте другой.")

st.caption("Данные: датасет с Medicine Name, Uses и Review %")