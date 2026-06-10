# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("=" * 50)
print("🐱 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ КОШЕК")
print("=" * 50)

# 1. Загружаем данные
print("\n1. Загрузка данных...")
try:
    df = pd.read_csv('cat_2.csv', sep=';')
    print(f"   ✅ Загружено {len(df)} записей")
    print(f"   📊 Колонки: {list(df.columns)[:10]}...")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 2. Создаем целевую переменную 'Healthy'
print("\n2. Создание целевой переменной 'Healthy'...")

def determine_health(row):
    score = 0
    # Играет достаточно (> 25 минут)
    if row['играет (мин.)'] > 25:
        score += 1
    # Спит нормально (12-18 часов)
    if 12 <= row['спит (часы)'] <= 18:
        score += 1
    # Вес в норме (2-8 кг)
    if 2 <= row['вес'] <= 8:
        score += 1
    # Возраст до 15 лет
    if row['возраст'] <= 15:
        score += 1
    return 'Yes' if score >= 3 else 'No'

df['Healthy'] = df.apply(determine_health, axis=1)

healthy_counts = df['Healthy'].value_counts()
print(f"   Здоровые (Yes): {healthy_counts.get('Yes', 0)}")
print(f"   Не здоровые (No): {healthy_counts.get('No', 0)}")

# 3. Готовим признаки
print("\n3. Подготовка признаков...")
feature_columns = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']
X = df[feature_columns]
y = df['Healthy']

# Заполняем пропуски
X = X.fillna(X.mean())
print(f"   Признаки: {feature_columns}")

# 4. Кодируем целевую переменную
print("\n4. Кодирование целевой переменной...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Обучаем модель
print("\n5. Обучение модели Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка качества
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"   Точность на обучении: {train_score:.2f}")
print(f"   Точность на тесте: {test_score:.2f}")

# 6. Сохраняем модель
print("\n6. Сохранение модели...")
with open('dog_health_model.pkl', 'wb') as f:
    pickle.dump((model, le, feature_columns), f)

# Проверяем, что файл создан
if os.path.exists('dog_health_model.pkl'):
    file_size = os.path.getsize('dog_health_model.pkl')
    print(f"   ✅ Модель сохранена: dog_health_model.pkl ({file_size} bytes)")
else:
    print("   ❌ Ошибка сохранения!")

print("\n" + "=" * 50)
print("🎉 ГОТОВО! Теперь запустите: streamlit run app.py")
print("=" * 50)
