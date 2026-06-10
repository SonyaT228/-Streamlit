# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("=" * 60)
print("🐱 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ КОШЕК")
print("=" * 60)

# 1. Загружаем данные
print("\n1. Загрузка данных...")
df = pd.read_csv('cat_2.csv', sep=';')
print(f"   Загружено {len(df)} записей")

# 2. Создаем целевую переменную Healthy
print("\n2. Создание целевой переменной...")

def is_healthy(row):
    """Определяет, здорова ли кошка"""
    problems = 0
    
    # Если кошка старая (> 15 лет) - проблемы
    if row['возраст'] > 15:
        problems += 1
    
    # Если вес слишком маленький или большой
    if row['вес'] < 2 or row['вес'] > 8:
        problems += 1
    
    # Если мало играет (< 20 минут)
    if row['играет (мин.)'] < 20:
        problems += 1
    
    # Если слишком много или мало спит
    if row['спит (часы)'] < 10 or row['спит (часы)'] > 20:
        problems += 1
    
    return 'No' if problems >= 2 else 'Yes'

df['Healthy'] = df.apply(is_healthy, axis=1)

print(f"   Здоровые (Yes): {sum(df['Healthy'] == 'Yes')}")
print(f"   Не здоровые (No): {sum(df['Healthy'] == 'No')}")

# 3. Выбираем признаки
print("\n3. Подготовка признаков...")
feature_columns = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']
X = df[feature_columns]
y = df['Healthy']

# Заполняем возможные пропуски
X = X.fillna(X.mean())
print(f"   Признаки: {feature_columns}")

# 4. Кодируем целевую переменную
print("\n4. Кодирование...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Обучаем модель
print("\n5. Обучение модели...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка
accuracy = model.score(X_test, y_test)
print(f"   Точность модели: {accuracy:.2f}")

# 6. Сохраняем модель
print("\n6. Сохранение модели...")
with open('dog_health_model.pkl', 'wb') as f:
    pickle.dump((model, le, feature_columns), f)

# Проверяем, создался ли файл
if os.path.exists('dog_health_model.pkl'):
    size = os.path.getsize('dog_health_model.pkl')
    print(f"   ✅ Модель сохранена! (файл: {size} байт)")
    print(f"   📁 Путь: {os.path.abspath('dog_health_model.pkl')}")
else:
    print("   ❌ Ошибка: файл не создан!")

print("\n" + "=" * 60)
print("✅ ГОТОВО! Теперь запустите: streamlit run app.py")
print("=" * 60)
