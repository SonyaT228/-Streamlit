# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

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
            model_data = pickle.load(f)
        
        # Поддержка двух форматов сохранения
        if isinstance(model_data, tuple):
            model, label_encoder, feature_columns = model_data
            scaler = None
        else:
            model = model_data['model']
            label_encoder = model_data['label_encoder']
            feature_columns = model_data['feature_columns']
            scaler = model_data.get('scaler', None)
        
        return model, label_encoder, feature_columns, scaler
    except FileNotFoundError:
        st.error("❌ Модель не найдена! Сначала запустите `python train_model.py`")
        return None, None, None, None

# Загружаем данные и модель
df = load_data()
model, label_encoder, feature_cols, scaler = load_model()

# --- СОЗДАНИЕ СТАТУСА ЗДОРОВЬЯ ДЛЯ ОТОБРАЖЕНИЯ ---
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

# --- БОКОВАЯ ПАНЕЛЬ С КОНТРОЛАМИ ---
st.sidebar.header("🔧 Панель управления")

# Контрол 1: Мультивыбор пород
breeds = st.sidebar.multiselect(
    "🐈 Выберите породы:",
    options=sorted(df['порода'].dropna().unique()),
    default=[]
)

# Контрол 2: Слайдер для веса
min_weight = float(df['вес'].min())
max_weight = float(df['вес'].max())
weight_range = st.sidebar.slider(
    "⚖️ Диапазон веса (кг):",
    min_value=min_weight,
    max_value=max_weight,
    value=(min_weight, max_weight),
    step=0.5
)

# Контрол 3: Выбор цвета шерсти
colors = st.sidebar.multiselect(
    "🎨 Цвет шерсти:",
    options=sorted(df['цвета'].dropna().unique()),
    default=[]
)

# Контрол 4: Чекбокс "Только здоровые"
only_healthy = st.sidebar.checkbox("✅ Показать только здоровых кошек")

# Контрол 5: Радио-кнопка для типа графика
plot_type = st.sidebar.radio(
    "📊 Тип графика:",
    options=["Гистограмма пород", "Средний вес по породам", "Активность по породам"]
)

# Контрол 6: Выбор пола (дополнительный)
gender_filter = st.sidebar.selectbox(
    "🚻 Пол:",
    options=["Все", "female", "male"]
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
if gender_filter != "Все":
    filtered_df = filtered_df[filtered_df['пол'] == gender_filter]

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Исходные данные")
st.info(f"📋 Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True, height=400)

# --- ВИЗУАЛИЗАЦИЯ ---
st.header("📈 Визуализация")

if plot_type == "Гистограмма пород":
    breed_counts = filtered_df['порода'].value_counts().head(10)
    st.subheader("Распределение пород")
    st.bar_chart(breed_counts)
    
    # Показываем числовые значения
    st.caption("Количество кошек по породам:")
    for breed, count in breed_counts.items():
        st.write(f"- {breed}: {count}")

elif plot_type == "Средний вес по породам":
    avg_weight = filtered_df.groupby('порода')['вес'].mean().sort_values(ascending=False).head(10)
    st.subheader("Средний вес по породам (кг)")
    st.bar_chart(avg_weight)
    
    # Показываем числовые значения
    st.caption("Средний вес:")
    for breed, weight in avg_weight.items():
        st.write(f"- {breed}: {weight:.1f} кг")

else:  # Активность по породам
    avg_play = filtered_df.groupby('порода')['играет (мин.)'].mean().sort_values(ascending=False).head(10)
    st.subheader("Среднее время игр по породам (минут в день)")
    st.bar_chart(avg_play)
    
    # Показываем числовые значения
    st.caption("Среднее время игр:")
    for breed, play in avg_play.items():
        st.write(f"- {breed}: {play:.0f} мин/день")

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
        # Подготавливаем данные для предсказания
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], 
                                  columns=feature_cols)
        
        # Применяем scaler если он есть
        if scaler:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
            probabilities = model.predict_proba(input_data_scaled)[0]
        else:
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
        
        recommendations = []
        if age > 12:
            recommendations.append("📌 Кошка пожилого возраста. Рекомендуется регулярный осмотр у ветеринара (2 раза в год).")
        if weight < 2:
            recommendations.append("📌 Низкий вес. Проконсультируйтесь с ветеринаром о питании.")
        if weight > 8:
            recommendations.append("📌 Избыточный вес. Рекомендуется диета и увеличение физической активности.")
        if play_mins < 20:
            recommendations.append("📌 Низкая активность. Попробуйте увеличить время игр с кошкой.")
        if sleep_hrs < 10:
            recommendations.append("📌 Мало спит. Возможно, кошке что-то мешает. Обеспечьте спокойное место для сна.")
        if sleep_hrs > 20:
            recommendations.append("📌 Спит слишком много. Обратите внимание на активность кошки.")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✨ Все параметры в норме! Продолжайте в том же духе.")

# --- ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА ---
st.markdown("---")
st.markdown("### 📊 Статистика по данным")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Всего кошек", len(df))

with col2:
    unique_breeds = df['порода'].nunique()
    st.metric("Пород", unique_breeds)

with col3:
    avg_weight = df['вес'].mean()
    st.metric("Средний вес", f"{avg_weight:.1f} кг")

with col4:
    avg_play = df['играет (мин.)'].mean()
    st.metric("Среднее время игр", f"{avg_play:.0f} мин")

with col5:
    healthy_count = len(df[df['Статус здоровья'] == 'Здорова'])
    healthy_pct = (healthy_count / len(df)) * 100
    st.metric("Здоровых кошек", f"{healthy_count} ({healthy_pct:.0f}%)")

# --- ТОП-5 САМЫХ АКТИВНЫХ КОШЕК ---
st.markdown("---")
st.markdown("### 🏆 Самые активные кошки")

top_active = df.nlargest(5, 'играет (мин.)')[['порода', 'возраст', 'вес', 'играет (мин.)', 'спит (часы)', 'пол']]
st.dataframe(top_active, use_container_width=True)

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
    
    **Автор:** Учебный проект по Streamlit
    """)

# --- ФУТЕР ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🐱 Cat Health Predictor | Сделано с ❤️ для заботливых владельцев кошек</p>", 
    unsafe_allow_html=True
)
