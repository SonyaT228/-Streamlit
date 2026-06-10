# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    return df

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    try:
        with open('dog_health_model.pkl', 'rb') as f:
            model, label_encoder, feature_columns = pickle.load(f)
        return model, label_encoder, feature_columns
    except FileNotFoundError:
        st.error("❌ Модель не найдена! Сначала запустите `python train_model.py`")
        return None, None, None

# Загружаем данные и модель
df = load_data()
model, label_encoder, feature_cols = load_model()

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ ---
st.sidebar.header("🔧 Панель управления")

# Контрол 1: Мультивыбор пород
breeds = st.sidebar.multiselect(
    "🐈 Выберите породы:",
    options=df['порода'].dropna().unique(),
    default=[]
)

# Контрол 2: Слайдер для веса
min_weight = float(df['вес'].min())
max_weight = float(df['вес'].max())
weight_range = st.sidebar.slider(
    "⚖️ Диапазон веса (кг):",
    min_value=min_weight,
    max_value=max_weight,
    value=(min_weight, max_weight)
)

# Контрол 3: Выбор цвета шерсти
colors = st.sidebar.multiselect(
    "🎨 Цвет шерсти:",
    options=df['цвета'].dropna().unique(),
    default=[]
)

# Контрол 4: Чекбокс "Только здоровые" (на основе созданной логики)
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

only_healthy = st.sidebar.checkbox("✅ Показать только здоровых кошек")

# Контрол 5: Радио-кнопка для типа графика
plot_type = st.sidebar.radio(
    "📊 Тип графика:",
    options=["Гистограмма пород", "Средний вес по породам", "Активность по породам"]
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
st.info(f"📋 Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# --- ВИЗУАЛИЗАЦИЯ ---
st.header("📈 Визуализация")

if plot_type == "Гистограмма пород":
    breed_counts = filtered_df['порода'].value_counts().head(10)
    st.bar_chart(breed_counts)
    
elif plot_type == "Средний вес по породам":
    avg_weight = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_weight)
    
else:  # Активность по породам
    avg_play = filtered_df.groupby('порода')['играет (мин.)'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_play)

# --- ПРОГНОЗИРОВАНИЕ ЗДОРОВЬЯ ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown("Введите данные о кошке, чтобы предсказать, будет ли она здорова.")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "📅 Возраст (лет)", 
            min_value=0.0, 
            max_value=30.0, 
            value=3.0,
            step=0.5,
            help="Средняя продолжительность жизни кошек 12-15 лет"
        )
        
        weight = st.number_input(
            "⚖️ Вес (кг)", 
            min_value=1.0, 
            max_value=15.0, 
            value=4.0,
            step=0.5,
            help="Нормальный вес для большинства пород: 3-5 кг"
        )
    
    with col2:
        play_mins = st.number_input(
            "🎾 Время игр (минут в день)", 
            min_value=0, 
            max_value=120, 
            value=30,
            step=5,
            help="Рекомендуется 30-60 минут активных игр"
        )
        
        sleep_hrs = st.number_input(
            "😴 Часы сна в день", 
            min_value=5.0, 
            max_value=24.0, 
            value=14.0,
            step=0.5,
            help="Кошки спят 12-18 часов в сутки"
        )
    
    # Кнопка предсказания
    if st.button("🔮 Предсказать здоровье", type="primary", use_container_width=True):
        # Формируем данные для предсказания
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], 
                                  columns=feature_cols)
        
        # Делаем предсказание
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Преобразуем результат
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.markdown("---")
        st.subheader("📋 Результат прогноза:")
        
        # Отображаем результат
        col_result, col_prob = st.columns(2)
        
        with col_result:
            if result == "Yes":
                st.success("✅ **Кошка, скорее всего, ЗДОРОВА**")
                st.balloons()
            else:
                st.error("⚠️ **Кошка, возможно, НЕЗДОРОВА**")
        
        with col_prob:
            if result == "Yes":
                st.metric("Вероятность здоровья", f"{probabilities[1]*100:.1f}%")
            else:
                st.metric("Вероятность проблем", f"{probabilities[0]*100:.1f}%")
        
        # Дополнительные рекомендации
        st.markdown("---")
        st.markdown("### 💡 Рекомендации:")
        
        if age > 12:
            st.info("📌 Кошка пожилого возраста. Рекомендуется регулярный осмотр у ветеринара (2 раза в год).")
        if weight < 2:
            st.warning("📌 Низкий вес. Проконсультируйтесь с ветеринаром о питании.")
        if weight > 8:
            st.warning("📌 Избыточный вес. Рекомендуется диета и увеличение физической активности.")
        if play_mins < 20:
            st.info("📌 Низкая активность. Попробуйте увеличить время игр с кошкой.")
        if sleep_hrs < 10:
            st.info("📌 Мало спит. Возможно, кошке что-то мешает. Обеспечьте спокойное место для сна.")
        if sleep_hrs > 20:
            st.info("📌 Спит слишком много. Обратите внимание на активность кошки.")

# --- ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ---
st.markdown("---")
st.markdown("### 📊 Статистика по данным")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Всего кошек", len(df))

with col2:
    unique_breeds = df['порода'].nunique()
    st.metric("Пород", unique_breeds)

with col3:
    avg_weight = df['вес'].mean()
    st.metric("Средний вес", f"{avg_weight:.1f} кг")

with col4:
    healthy_count = len(df[df['Статус здоровья'] == 'Здорова'])
    healthy_pct = (healthy_count / len(df)) * 100
    st.metric("Здоровых кошек", f"{healthy_count} ({healthy_pct:.0f}%)")

# --- ИНФОРМАЦИЯ О ПРОЕКТЕ ---
with st.expander("ℹ️ О проекте"):
    st.markdown("""
    **Cat Health Predictor** - приложение для анализа и прогнозирования здоровья кошек.
    
    **Как это работает:**
    - Модель обучена на данных о кошках трех пород: Ангорская, Мейн-кун, Рэгдолл
    - Для предсказания используются: возраст, вес, время игр и продолжительность сна
    - Точность модели: ~85%
    
    **Источник данных:**
    - Породы, возраст, вес, активность, режим сна, страна проживания
    
    **Технологии:**
    - Streamlit - веб-интерфейс
    - Pandas - обработка данных
    - Scikit-learn - машинное обучение (Random Forest)
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🐱 Cat Health Predictor | Сделано с ❤️ для заботливых владельцев кошек</p>", 
    unsafe_allow_html=True
)
