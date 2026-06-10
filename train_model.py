import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1️⃣ ЗАГРУЗКА ДАННЫХ
df = pd.read_csv("cat_2.csv", sep=';')

# 2️⃣ ПРИВЕДЕНИЕ НАЗВАНИЙ КОЛОНОК К СТАНДАРТУ APP.PY
# Переименовываем колонки из CSV в понятные названия
rename_dict = {
    "возраст": "Age",
    "вес": "Weight",
    "играет (мин.)": "Play Minutes",
    "спит (часы)": "Sleep Hours",
    "порода": "Breed",
    "пол": "Gender",
    "сетрелизация": "Fixed",
    "цвета": "Color",
    "гуляет или нет": "Outdoor",
    "еда": "Food",
    "страна": "Country"
}
df = df.rename(columns=rename_dict)

# 3️⃣ ПОДГОТОВКА ДАННЫХ
# Создаем целевую переменную 'Healthy' на основе критериев здоровья кошки
def is_healthy(row):
    problems = 0
    if row['Age'] > 15:
        problems += 1
    if row['Weight'] < 2 or row['Weight'] > 8:
        problems += 1
    if row['Play Minutes'] < 20:
        problems += 1
    if row['Sleep Hours'] < 10 or row['Sleep Hours'] > 20:
        problems += 1
    return 'No' if problems >= 2 else 'Yes'

df["Healthy"] = df.apply(is_healthy, axis=1)
y = df["Healthy"]  # Это то, что будем предсказывать

# Выбираем признаки (названия СТРОГО СОВПАДАЮТ с app.py)
feature_columns = [
    "Age",
    "Weight",
    "Play Minutes",
    "Sleep Hours"
]
X = df[feature_columns]

# Заполняем пропуски в данных средними значениями
X = X.fillna(X.mean())

print(f"Распределение Healthy:\n{df['Healthy'].value_counts()}")

# 4️⃣ КОДИРОВАНИЕ
# Превращаем 'Yes'/'No' в 1/0 для математических расчетов
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Yes' -> 1, 'No' -> 0

# 5️⃣ ОБУЧЕНИЕ МОДЕЛИ
# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Создаем и обучаем модель (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Здесь происходит само обучение!

# Проверяем качество модели
print(f"Модель обучена. Точность на тесте: {model.score(X_test, y_test):.2f}")

# Дополнительно: важность признаков
print("\nВажность признаков:")
for col, imp in zip(feature_columns, model.feature_importances_):
    print(f"  {col}: {imp:.3f}")

# 6️⃣ СОХРАНЕНИЕ МОДЕЛИ
# Сохраняем модель, кодировщик и список колонок в файл для последующего использования
with open("dog_health_model.pkl", "wb") as f:
    pickle.dump((model, le, feature_columns), f)

print("\nМодель сохранена в файл 'dog_health_model.pkl'")
