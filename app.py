# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ (Задание 1) ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    
    # Создаем метку здоровья для отображения
    def get_health_status(row):
        score = 0
        if row['играет (мин.)'] > 25:
            score += 1
        if 12 <= row['спит (часы)'] <= 18:
            score += 1
        if 2 <= row['вес'] <= 8:
            score += 1
        if row['возраст'] <= 15:
            score += 1
        return 'Здорова' if score >= 3 else 'Требует внимания'
    
    df['Статус здоровья'] = df.apply(get_health_status, axis=1)
    return df

# --- ЗАГРУЗКА МОДЕЛИ (тот же файл, что и в примере - dog_health_model.pkl) ---
@st.cache_resource
def load_model():
    try:
        with open('dog_health_model.pkl', 'rb') as f:
            model, label_encoder, feature_columns = pickle.load(f)
        return model, label_encoder, feature_columns
    except FileNotFoundError:
        st.error("❌ Модель не найдена! Сначала запустите `python train_model.py`")
        return None, None, None

df = load_data()
model, label_encoder, feature_cols = load_model()

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ ---
st.sidebar.header("🔧 Панель управления")

# 1. Мультивыбор пород
breeds = st.sidebar.multiselect(
    "Выберите породы:",
    options=df['порода'].dropna().unique(),
    default=[]
)

# 2. Слайдер для веса
weight_range = st.sidebar.slider(
    "Диапазон веса (кг):",
    min_value=float(df['вес'].min()),
    max_value=float(df['вес'].max()),
    value=(float(df['вес'].min()), float(df['вес'].max()))
)

# 3. Выбор цвета шерсти
colors = st.sidebar.multiselect(
    "Цвет шерсти:",
    options=df['цвета'].dropna().unique(),
    default=[]
)

# 4. Чекбокс "Только здоровые"
only_healthy = st.sidebar.checkbox("Показать только здоровых кошек")

# 5. Радио-кнопка для типа графика
plot_type = st.sidebar.radio(
    "Тип графика:",
    options=["Гистограмма пород", "Средний вес по породам"]
)

# Применяем фильтры
filtered_df = df.copy()
if breeds:
    filtered_df = filtered_df[filtered_df['порода'].isin(breeds)]
filtered_df = filtered_df[(filtered_df['вес'] >= weight_range[0]) & 
                          (filtered_df['вес'] <= weight_range[1])]
if colors:
    filtered_df = filtered_df[filtered_df['цвета'].isin(colors)]
if only_healthy:
    filtered_df = filtered_df[filtered_df['Статус здоровья'] == 'Здорова']

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Исходные данные")
st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# Визуализация
st.header("📈 Визуализация")
if plot_type == "Гистограмма пород":
    breed_counts = filtered_df['порода'].value_counts().head(10)
    st.bar_chart(breed_counts)
else:
    avg_weight = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_weight)

# --- МОДЕЛЬ МАШИННОГО ОБУЧЕНИЯ (Задание 2) ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown("Введите данные о кошке, чтобы предсказать, будет ли она здорова.")

if model is not None:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0)
        weight = st.number_input("Вес (кг)", min_value=1.0, max_value=15.0, value=4.0)
    with col2:
        play_mins = st.number_input("Время игр (мин.)", min_value=0, max_value=120, value=30)
        sleep_hrs = st.number_input("Часы сна", min_value=5.0, max_value=24.0, value=14.0)

    if st.button("🔮 Предсказать здоровье", type="primary"):
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], 
                                  columns=feature_cols)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Результат прогноза:")
        if result == "Yes":
            st.success(f"✅ Кошка, скорее всего, ЗДОРОВА (Вероятность: {probability[1]:.2f})")
        else:
            st.error(f"❌ Кошка, возможно, НЕЗДОРОВА (Вероятность проблем: {probability[0]:.2f})")
