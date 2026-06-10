# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    return df

# --- ФУНКЦИЯ ДЛЯ ОПРЕДЕЛЕНИЯ ЗДОРОВЬЯ ---
def is_healthy(row):
    problems = 0
    if row['возраст'] > 15:
        problems += 1
    if row['вес'] < 2 or row['вес'] > 8:
        problems += 1
    if row['играет (мин.)'] < 20:
        problems += 1
    if row['спит (часы)'] < 10 or row['спит (часы)'] > 20:
        problems += 1
    return 'Yes' if problems < 2 else 'No'

# --- ЗАГРУЗКА И ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def get_model():
    df = load_data()
    
    # Создаем целевую переменную
    df['Healthy'] = df.apply(is_healthy, axis=1)
    
    # Признаки
    feature_cols = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Healthy']
    
    # Кодирование
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    return model, le, feature_cols

# --- ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---
try:
    # Загружаем данные
    df = load_data()
    
    # Получаем модель
    model, label_encoder, feature_cols = get_model()
    
    # Добавляем статус здоровья для отображения
    df['Статус'] = df.apply(is_healthy, axis=1).map({'Yes': 'Здорова', 'No': 'Требует внимания'})
    
    # --- БОКОВАЯ ПАНЕЛЬ ---
    st.sidebar.header("🔧 Панель управления")
    
    # Фильтр по породе
    breeds = st.sidebar.multiselect(
        "Выберите породы:",
        options=sorted(df['порода'].unique()),
        default=[]
    )
    
    # Фильтр по весу
    min_w = float(df['вес'].min())
    max_w = float(df['вес'].max())
    weight_range = st.sidebar.slider(
        "Диапазон веса (кг):",
        min_value=min_w,
        max_value=max_w,
        value=(min_w, max_w)
    )
    
    # Фильтр по цвету
    colors = st.sidebar.multiselect(
        "Цвет шерсти:",
        options=sorted(df['цвета'].unique()),
        default=[]
    )
    
    # Чекбокс "только здоровые"
    only_healthy = st.sidebar.checkbox("Показать только здоровых кошек")
    
    # Тип графика
    plot_type = st.sidebar.radio(
        "Тип графика:",
        ["Гистограмма пород", "Средний вес по породам"]
    )
    
    # Применяем фильтры
    filtered_df = df.copy()
    if breeds:
        filtered_df = filtered_df[filtered_df['порода'].isin(breeds)]
    filtered_df = filtered_df[(filtered_df['вес'] >= weight_range[0]) & (filtered_df['вес'] <= weight_range[1])]
    if colors:
        filtered_df = filtered_df[filtered_df['цвета'].isin(colors)]
    if only_healthy:
        filtered_df = filtered_df[filtered_df['Статус'] == 'Здорова']
    
    # --- ОСНОВНАЯ ОБЛАСТЬ ---
    st.header("📊 Исходные данные")
    st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
    st.dataframe(filtered_df, use_container_width=True)
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    st.header("📈 Визуализация")
    if plot_type == "Гистограмма пород":
        data = filtered_df['порода'].value_counts()
        st.bar_chart(data)
    else:
        data = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False)
        st.bar_chart(data)
    
    # --- ПРОГНОЗИРОВАНИЕ ---
    st.header("🤖 Прогнозирование здоровья кошки")
    st.markdown("Введите данные о кошке:")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0)
        weight = st.number_input("Вес (кг)", min_value=1.0, max_value=15.0, value=4.0)
    with col2:
        play_mins = st.number_input("Время игр (минут в день)", min_value=0, max_value=120, value=30)
        sleep_hrs = st.number_input("Часы сна в день", min_value=5.0, max_value=24.0, value=14.0)
    
    if st.button("🔮 Предсказать здоровье", type="primary"):
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], columns=feature_cols)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Результат прогноза:")
        if result == "Yes":
            st.success(f"✅ Кошка ЗДОРОВА! (вероятность: {prob[1]*100:.1f}%)")
            st.balloons()
        else:
            st.error(f"⚠️ Кошка требует внимания! (вероятность проблем: {prob[0]*100:.1f}%)")
    
    # --- СТАТИСТИКА ---
    st.markdown("---")
    st.markdown("### 📊 Статистика")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего кошек", len(df))
    with col2:
        st.metric("Пород", df['порода'].nunique())
    with col3:
        st.metric("Средний вес", f"{df['вес'].mean():.1f} кг")
    with col4:
        healthy_count = sum(df['Статус'] == 'Здорова')
        st.metric("Здоровых", f"{healthy_count} ({healthy_count/len(df)*100:.0f}%)")

except Exception as e:
    st.error(f"Ошибка: {e}")
    st.info("Проверьте, что файл 'cat_2.csv' находится в той же папке")
