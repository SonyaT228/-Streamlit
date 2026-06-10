# app.py (обновленная версия с загрузкой модели)
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
    # Создаем целевую переменную для отображения
    df['Healthy'] = ((df['играет (мин.)'] > 30) & 
                     (df['спит (часы)'] >= 12) & 
                     (df['спит (часы)'] <= 18)).map({True: 'Yes', False: 'No'})
    return df

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_trained_model():
    try:
        with open('cat_health_model.pkl', 'rb') as f:
            model, label_encoder, feature_columns = pickle.load(f)
        return model, label_encoder, feature_columns
    except FileNotFoundError:
        st.error("Файл модели 'cat_health_model.pkl' не найден. Запустите train_model.py для обучения модели.")
        return None, None, None

df = load_data()
model, label_encoder, feature_cols = load_trained_model()

# ... (остальной код интерфейса остается таким же)
