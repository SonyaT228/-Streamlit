# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Загружаем данные (обратите внимание на разделитель ';')
df = pd.read_csv('cat_2.csv', sep=';')

# 2. Создаем целевую переменную 'Healthy' на основе имеющихся данных
# Критерии здоровья кошки (можно настроить под ваши задачи):
# - Нормальный сон (14-18 часов - норма для кошек)
# - Достаточная активность (играет более 30 минут в день)
# - Нормальный вес (2-6 кг для большинства пород)
# - Отсутствие проблем (можно добавить другие критерии)

df['Healthy'] = (
    (df['спит (часы)'] >= 12) & 
    (df['спит (часы)'] <= 18) & 
    (df['играет (мин.)'] >= 20)
).astype(int)

# Преобразуем в Yes/No для удобства
df['Healthy'] = df['Healthy'].map({1: 'Yes', 0: 'No'})

# Альтернативный вариант: если в данных есть другие признаки здоровья
# Например, можно использовать комбинацию возраста и активности
# df['Healthy'] = ((df['возраст'] < 15) & (df['играет (мин.)'] > 25)).map({True: 'Yes', False: 'No'})

print(f"Распределение целевой переменной:\n{df['Healthy'].value_counts()}")

# 3. Готовим признаки (X) и целевую переменную (y)
y = df['Healthy']

# Выбираем числовые признаки для модели
feature_columns = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']

# Можно добавить категориальные признаки, закодировав их
# Например, пол и стерилизацию:
# from sklearn.preprocessing import LabelEncoder
# le_gender = LabelEncoder()
# df['пол_encoded'] = le_gender.fit_transform(df['пол'])
# feature_columns.append('пол_encoded')

X = df[feature_columns]

# Проверяем и заполняем пропуски
print(f"Пропуски в признаках:\n{X.isnull().sum()}")
X = X.fillna(X.mean())

# 4. Кодируем целевую переменную (Yes -> 1, No -> 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Классы: {le.classes_}")

# 5. Обучаем модель
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Оценка качества
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Точность на обучении: {train_score:.2f}")
print(f"Точность на тесте: {test_score:.2f}")

# Важность признаков
feature_importance = pd.DataFrame({
    'Признак': feature_columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков:")
print(feature_importance)

# 6. Сохраняем модель и кодировщик цели
with open('cat_health_model.pkl', 'wb') as f:
    pickle.dump((model, le, feature_columns), f)

print("\n✅ Модель сохранена в файл 'cat_health_model.pkl'")
