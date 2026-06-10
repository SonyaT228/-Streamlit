# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# --- ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ ---
if not os.path.exists('dog_health_model.pkl'):
    st.error("""
    ❌ **Модель не найдена!**
    
    Пожалуйста, выполните следующие шаги:
    
    1. Откройте терминал/командную строку
    2. Перейдите в папку с проектом
    3. Запустите команду: **python train_model.py**
    4. После успешного обучения перезапустите это приложение
    
    ---
    **Текущая директория:** `{}`
    **Файлы в директории:** {}
    """.format(
        os.getcwd(),
        ', '.join(os.listdir()) if os.listdir() else 'нет файлов'
    ))
    st.stop()

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    return df

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    with open('dog_health_model.pkl', 'rb') as f:
        model, label_encoder, feature_columns = pickle.load(f)
    return model, label_encoder, feature_columns

# Загружаем данные и модель
df = load_data()
model, label_encoder, feature_cols = load_model()

# --- СОЗДАЕМ СТАТУС ЗДОРОВЬЯ ДЛЯ ОТОБРАЖЕНИЯ ---
def get_status(row):
    problems = 0
    if row['возраст'] > 15:
        problems += 1
    if row['вес'] < 2 or row['вес'] > 8:
        problems += 1
    if row['играет (мин.)'] < 20:
        problems += 1
    if row['спит (часы)'] < 10 or row['спит (часы)'] > 20:
        problems += 1
    return 'Здорова' if problems < 2 else 'Требует внимания'

df['Статус здоровья'] = df.apply(get_status, axis=1)

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("🔧 Панель управления")

# Фильтр по породе
breeds = st.sidebar.multiselect(
    "Выберите породы:",
    options=sorted(df['порода'].unique()),
    default=[]
)

# Фильтр по весу
weight_range = st.sidebar.slider(
    "Диапазон веса (кг):",
    min_value=float(df['вес'].min()),
    max_value=float(df['вес'].max()),
    value=(float(df['вес'].min()), float(df['вес'].max()))
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
    ["Гистограмма пород", "Средний вес по породам", "Активность по породам"]
)

# Фильтр по полу
gender = st.sidebar.selectbox("Пол:", ["Все", "female", "male"])

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
if gender != "Все":
    filtered_df = filtered_df[filtered_df['пол'] == gender]

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Исходные данные")
st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# --- ВИЗУАЛИЗАЦИЯ ---
st.header("📈 Визуализация")

if plot_type == "Гистограмма пород":
    data = filtered_df['порода'].value_counts()
    st.bar_chart(data)
    
elif plot_type == "Средний вес по породам":
    data = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False)
    st.bar_chart(data)
    
else:  # Активность по породам
    data = filtered_df.groupby('порода')['играет (мин.)'].mean().sort_values(ascending=False)
    st.bar_chart(data)

# --- ПРОГНОЗИРОВАНИЕ ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown("Введите данные о кошке:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    weight = st.number_input("Вес (кг)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)

with col2:
    play_mins = st.number_input("Время игр (мин. в день)", min_value=0, max_value=120, value=30, step=5)
    sleep_hrs = st.number_input("Часы сна в день", min_value=5.0, max_value=24.0, value=14.0, step=0.5)

if st.button("🔮 Предсказать здоровье", type="primary"):
    # Подготавливаем данные
    input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], 
                              columns=feature_cols)
    
    # Предсказываем
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    result = label_encoder.inverse_transform([prediction])[0]
    
    st.markdown("---")
    st.subheader("Результат прогноза:")
    
    if result == "Yes":
        st.success(f"✅ Кошка ЗДОРОВА! (вероятность: {probability[1]*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"⚠️ Кошка требует внимания! (вероятность проблем: {probability[0]*100:.1f}%)")

# --- СТАТИСТИКА ---
st.markdown("---")
st.markdown("### 📊 Общая статистика")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Всего кошек", len(df))
with col2:
    st.metric("Пород", df['порода'].nunique())
with col3:
    st.metric("Средний вес", f"{df['вес'].mean():.1f} кг")
with col4:
    healthy_count = sum(df['Статус здоровья'] == 'Здорова')
    st.metric("Здоровых кошек", f"{healthy_count} ({healthy_count/len(df)*100:.0f}%)") 
