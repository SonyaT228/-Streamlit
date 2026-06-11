# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

# --- ФУНКЦИЯ ДЛЯ ОПРЕДЕЛЕНИЯ ЗДОРОВЬЯ ---
def is_healthy(row):
    problems = 0
    if row['Age'] > 15:
        problems += 1
    if row['Weight'] < 2 or row['Weight'] > 8:
        problems += 1
    if row['Playing (min.)'] < 20:
        problems += 1
    if row['Sleeps (hours)'] < 10 or row['Sleeps (hours)'] > 20:
        problems += 1
    return 'Yes' if problems < 2 else 'No'

# --- ЗАГРУЗКА И ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def get_model():
    df = load_data()
    
    # Создаем целевую переменную
    df['Healthy'] = df.apply(is_healthy, axis=1)
    
    # Признаки
    feature_cols = ['Age', 'Weight', 'Playing (min.)', 'Sleeps (hours)']
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Healthy']
    
    # Кодирование
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    return model, le, feature_cols

# --- ОСНОВНАЯ ЛОГИКА ---
try:
    # Загружаем данные
    df = load_data()
    
    # Получаем модель
    model, label_encoder, feature_cols = get_model()
    
    # Добавляем статус здоровья для отображения
    df['Health Status'] = df.apply(is_healthy, axis=1).map({'Yes': 'Healthy', 'No': 'Needs Attention'})
    
    # --- БОКОВАЯ ПАНЕЛЬ ---
    st.sidebar.header("🔧 Filters")
    
    # Filter by breed
    breeds = st.sidebar.multiselect(
        "Select breeds:",
        options=sorted(df['Breed'].unique()),
        default=[]
    )
    
    # Filter by weight
    min_w = float(df['Weight'].min())
    max_w = float(df['Weight'].max())
    weight_range = st.sidebar.slider(
        "Weight range (kg):",
        min_value=min_w,
        max_value=max_w,
        value=(min_w, max_w)
    )
    
    # Filter by color
    colors = st.sidebar.multiselect(
        "Color:",
        options=sorted(df['Color'].unique()),
        default=[]
    )
    
    # Filter by gender
    gender_filter = st.sidebar.selectbox(
        "Gender:",
        options=["All", "female", "male"]
    )
    
    # Checkbox for healthy only
    only_healthy = st.sidebar.checkbox("Show only healthy cats")
    
    # Plot type
    plot_type = st.sidebar.radio(
        "Chart type:",
        ["Breed distribution", "Average weight by breed", "Activity by breed"]
    )
    
    # Apply filters
    filtered_df = df.copy()
    if breeds:
        filtered_df = filtered_df[filtered_df['Breed'].isin(breeds)]
    filtered_df = filtered_df[(filtered_df['Weight'] >= weight_range[0]) & (filtered_df['Weight'] <= weight_range[1])]
    if colors:
        filtered_df = filtered_df[filtered_df['Color'].isin(colors)]
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if only_healthy:
        filtered_df = filtered_df[filtered_df['Health Status'] == 'Healthy']
    
    # --- MAIN AREA ---
    st.header("📊 Data Overview")
    st.info(f"Showing: {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # --- VISUALIZATION ---
    st.header("📈 Charts")
    if plot_type == "Breed distribution":
        data = filtered_df['Breed'].value_counts()
        st.bar_chart(data)
    elif plot_type == "Average weight by breed":
        data = filtered_df.groupby('Breed')['Weight'].mean().sort_values(ascending=False)
        st.bar_chart(data)
    else:
        data = filtered_df.groupby('Breed')['Playing (min.)'].mean().sort_values(ascending=False)
        st.bar_chart(data)
    
    # --- PREDICTION ---
    st.header("🤖 Predict Cat Health")
    st.markdown("Enter your cat's data:")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=0.0, max_value=30.0, value=3.0)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=15.0, value=4.0)
    with col2:
        play_mins = st.number_input("Playing time (min/day)", min_value=0, max_value=120, value=30)
        sleep_hrs = st.number_input("Sleep hours per day", min_value=5.0, max_value=24.0, value=14.0)
    
    if st.button("🔮 Predict Health", type="primary"):
        input_data = pd.DataFrame([[age, weight, play_mins, sleep_hrs]], columns=feature_cols)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        
        st.subheader("Prediction Result:")
        if result == "Yes":
            st.success(f"✅ Cat is HEALTHY! (Probability: {prob[1]*100:.1f}%)")
            st.balloons()
        else:
            st.error(f"⚠️ Cat needs ATTENTION! (Risk probability: {prob[0]*100:.1f}%)")
    
    # --- STATISTICS ---
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total cats", len(df))
    with col2:
        st.metric("Breeds", df['Breed'].nunique())
    with col3:
        st.metric("Average weight", f"{df['Weight'].mean():.1f} kg")
    with col4:
        healthy_count = sum(df['Health Status'] == 'Healthy')
        st.metric("Healthy cats", f"{healthy_count} ({healthy_count/len(df)*100:.0f}%)")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure 'cat_2.csv' is in the same folder")
