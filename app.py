# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ (Задание 1) ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    
    # Создаем целевую переменную Healthy на основе критериев
    def determine_health(row):
        problems = 0
        if row['возраст'] > 15:
            problems += 1
        if row['вес'] < 2 or row['вес'] > 8:
            problems += 1
        if row['играет (мин.)'] < 20:
            problems += 1
        if row['спит (часы)'] < 10 or row['спит (часы)'] > 20:
            problems += 1
        return 'No' if problems >= 2 else 'Yes'
    
    df['Healthy'] = df.apply(determine_health, axis=1)
    return df

df = load_data()

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ ---
st.sidebar.header("🔧 Панель управления")

# Контрол 1: Мультивыбор пород
breeds = st.sidebar.multiselect(
    "Выберите породы для фильтрации:",
    options=df['порода'].dropna().unique(),
    default=[]
)

# Контрол 2: Слайдер для веса
weight_range = st.sidebar.slider(
    "Диапазон веса (кг):",
    min_value=float(df['вес'].min()),
    max_value=float(df['вес'].max()),
    value=(float(df['вес'].min()), float(df['вес'].max()))
)

# Контрол 3: Выбор цвета шерсти
color = st.sidebar.multiselect(
    "Цвет шерсти:",
    options=df['цвета'].dropna().unique(),
    default=[]
)

# Контрол 4: Чекбокс "Только здоровые"
only_healthy = st.sidebar.checkbox("Показать только здоровых кошек")

# Контрол 5: Радио-кнопка для типа графика
plot_type = st.sidebar.radio(
    "Тип графика для визуализации:",
    options=["Гистограмма пород", "Средний вес по породам", "Активность по породам"]
)

# Контрол 6: Выбор пола (дополнительный)
gender_filter = st.sidebar.selectbox(
    "Пол:",
    options=["Все", "female", "male"]
)

# Применяем фильтры
filtered_df = df.copy()
if breeds:
    filtered_df = filtered_df[filtered_df['порода'].isin(breeds)]
filtered_df = filtered_df[(filtered_df['вес'] >= weight_range[0]) & 
                          (filtered_df['вес'] <= weight_range[1])]
if color:
    filtered_df = filtered_df[filtered_df['цвета'].isin(color)]
if only_healthy:
    filtered_df = filtered_df[filtered_df['Healthy'] == 'Yes']
if gender_filter != "Все":
    filtered_df = filtered_df[filtered_df['пол'] == gender_filter]

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Исходные данные")
st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# Визуализация
st.header("📈 Визуализация")
if plot_type == "Гистограмма пород":
    breed_counts = filtered_df['порода'].value_counts().head(10)
    st.bar_chart(breed_counts)
elif plot_type == "Средний вес по породам":
    avg_weight = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_weight)
else:
    avg_play = filtered_df.groupby('порода')['играет (мин.)'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_play)

# --- МОДЕЛЬ МАШИННОГО ОБУЧЕНИЯ (Задание 2) ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown("Введите данные о кошке, чтобы предсказать, будет ли она здорова.")

# Модель обучается прямо здесь один раз и кэшируется
@st.cache_resource
def load_model():
    # Названия признаков, которые мы будем собирать с интерфейса
    feature_cols = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']
    
    # Подготовка матриц
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Healthy']
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Обучение модели Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)
    
    return model, label_encoder, feature_cols

# Вызов функции обучения
model, label_encoder, feature_cols = load_model()

# Интерфейс для ввода данных нового экземпляра
if model is not None:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0)
        weight = st.number_input("Вес (кг)", min_value=0.0, max_value=15.0, value=4.0)
        play_mins = st.number_input("Время игр (минут в день)", min_value=0, max_value=120, value=30)
    with col2:
        sleep_hrs = st.number_input("Часы сна", min_value=0.0, max_value=24.0, value=14.0)

    if st.button("🔮 Предсказать здоровье", type="primary"):
        # Формируем DataFrame с правильными именами колонок
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], 
                                  columns=feature_cols)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Результат прогноза:")
        if result == "Yes":
            st.success(f"✅ Кошка, скорее всего, ЗДОРОВА (Вероятность: {probability[1]:.2f})")
            st.balloons()
        else:
            st.error(f"❌ Кошка, возможно, НЕЗДОРОВА (Вероятность проблем: {probability[0]:.2f})")
            
        st.caption("Примечание: Прогноз основан на возрасте, весе, активности и режиме сна кошки.")
