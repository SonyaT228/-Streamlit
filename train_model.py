# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🐱 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПРЕДСКАЗАНИЯ ЗДОРОВЬЯ КОШЕК")
print("=" * 60)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
print("\n📂 Шаг 1: Загрузка данных...")

try:
    df = pd.read_csv('cat_2.csv', sep=';')
    print(f"   ✅ Успешно загружено {len(df)} записей")
    print(f"   📊 Колонки: {', '.join(df.columns[:8])}...")
except FileNotFoundError:
    print("   ❌ Ошибка: Файл 'cat_2.csv' не найден!")
    print("   Убедитесь, что файл находится в той же папке")
    exit(1)

# ============================================
# 2. АНАЛИЗ И ОЧИСТКА ДАННЫХ
# ============================================
print("\n🔧 Шаг 2: Анализ и очистка данных...")

# Проверяем пропуски
print(f"   Пропуски в данных:\n{df.isnull().sum()}")
print(f"   Типы данных:\n{df.dtypes}")

# Заполняем пропуски (если есть)
df = df.fillna({
    'возраст': df['возраст'].median(),
    'вес': df['вес'].median(),
    'играет (мин.)': df['играет (мин.)'].median(),
    'спит (часы)': df['спит (часы)'].median()
})

# ============================================
# 3. СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ 'Healthy'
# ============================================
print("\n🏥 Шаг 3: Создание целевой переменной 'Healthy'...")

def calculate_health_score(row):
    """
    Расчет здоровья кошки по шкале от 0 до 4
    """
    score = 0
    
    # Критерий 1: Возраст (молодые и взрослые кошки здоровее)
    if row['возраст'] <= 8:
        score += 1
    elif row['возраст'] <= 12:
        score += 0.5
    
    # Критерий 2: Вес (нормальный вес - признак здоровья)
    if 3 <= row['вес'] <= 6:
        score += 1
    elif 2 <= row['вес'] <= 8:
        score += 0.5
    
    # Критерий 3: Активность (кошки, которые много играют, здоровее)
    if row['играет (мин.)'] >= 30:
        score += 1
    elif row['играет (мин.)'] >= 20:
        score += 0.5
    
    # Критерий 4: Сон (нормальный режим сна)
    if 12 <= row['спит (часы)'] <= 16:
        score += 1
    elif 10 <= row['спит (часы)'] <= 18:
        score += 0.5
    
    return score

# Создаем score и целевую переменную
df['Health_Score'] = df.apply(calculate_health_score, axis=1)
df['Healthy'] = df['Health_Score'].apply(lambda x: 'Yes' if x >= 2.5 else 'No')

healthy_count = df['Healthy'].value_counts()
print(f"   ✅ Здоровые кошки (Yes): {healthy_count.get('Yes', 0)} ({healthy_count.get('Yes', 0)/len(df)*100:.1f}%)")
print(f"   ❌ Не здоровые (No): {healthy_count.get('No', 0)} ({healthy_count.get('No', 0)/len(df)*100:.1f}%)")
print(f"   📊 Средний балл здоровья: {df['Health_Score'].mean():.2f}")

# ============================================
# 4. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================
print("\n🔬 Шаг 4: Подготовка признаков для модели...")

# Числовые признаки
feature_columns = ['возраст', 'вес', 'играет (мин.)', 'спит (часы)']

# Создаем дополнительные признаки (для улучшения модели)
df['Age_Group'] = pd.cut(df['возраст'], bins=[0, 2, 8, 15, 30], labels=[0, 1, 2, 3])
df['Weight_Group'] = pd.cut(df['вес'], bins=[0, 2, 4, 6, 15], labels=[0, 1, 2, 3])
df['Activity_Ratio'] = df['играет (мин.)'] / df['спит (часы)']

# Добавляем дополнительные признаки в список
feature_columns.extend(['Age_Group', 'Weight_Group', 'Activity_Ratio'])

X = df[feature_columns].copy()
y = df['Healthy']

# Заполняем возможные пропуски
X = X.fillna(X.mean())

print(f"   ✅ Используемые признаки: {feature_columns}")
print(f"   📊 Размер матрицы признаков: {X.shape}")

# ============================================
# 5. КОДИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================
print("\n🔢 Шаг 5: Кодирование целевой переменной...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Yes' -> 1, 'No' -> 0

print(f"   ✅ Классы: {le.classes_} (0='No', 1='Yes')")
print(f"   📊 Распределение после кодирования: {np.bincount(y_encoded)}")

# ============================================
# 6. НОРМАЛИЗАЦИЯ ПРИЗНАКОВ
# ============================================
print("\n📊 Шаг 6: Нормализация признаков...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   ✅ Признаки нормализованы")

# ============================================
# 7. РАЗДЕЛЕНИЕ ДАННЫХ
# ============================================
print("\n✂️ Шаг 7: Разделение данных на обучающую и тестовую выборки...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"   ✅ Обучающая выборка: {len(X_train)} записей")
print(f"   ✅ Тестовая выборка: {len(X_test)} записей")

# ============================================
# 8. ОБУЧЕНИЕ МОДЕЛИ
# ============================================
print("\n🤖 Шаг 8: Обучение модели Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,           # Количество деревьев
    max_depth=10,               # Максимальная глубина дерева
    min_samples_split=5,        # Минимальное количество образцов для разделения узла
    min_samples_leaf=2,         # Минимальное количество образцов в листе
    random_state=42,            # Для воспроизводимости
    n_jobs=-1,                  # Используем все ядра процессора
    class_weight='balanced'     # Учитываем дисбаланс классов
)

model.fit(X_train, y_train)
print("   ✅ Модель обучена!")

# ============================================
# 9. ОЦЕНКА КАЧЕСТВА МОДЕЛИ
# ============================================
print("\n📈 Шаг 9: Оценка качества модели...")

# Предсказания
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Точность
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"   🎯 Точность на обучении: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
print(f"   🎯 Точность на тесте: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

# Детальный отчет
print("\n   📋 Classification Report (тестовая выборка):")
print(f"   {classification_report(y_test, y_test_pred, target_names=['No', 'Yes'])}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_test_pred)
print("   📊 Матрица ошибок:")
print(f"      Предсказано No  |  Предсказано Yes")
print(f"      {cm[0,0]:^15} | {cm[0,1]:^15}  (Реально No)")
print(f"      {cm[1,0]:^15} | {cm[1,1]:^15}  (Реально Yes)")

# ============================================
# 10. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================
print("\n🔍 Шаг 10: Анализ важности признаков...")

feature_importance = pd.DataFrame({
    'Признак': feature_columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\n   📊 Важность признаков (топ-5):")
for idx, row in feature_importance.head(5).iterrows():
    print(f"      {row['Признак']:20s}: {row['Важность']:.3f} ({row['Важность']*100:.1f}%)")

# ============================================
# 11. СОХРАНЕНИЕ МОДЕЛИ
# ============================================
print("\n💾 Шаг 11: Сохранение модели...")

# Сохраняем все необходимые компоненты
model_data = {
    'model': model,
    'label_encoder': le,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_accuracy': test_accuracy,
    'feature_importance': feature_importance.to_dict()
}

try:
    with open('dog_health_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("   ✅ Модель успешно сохранена в 'dog_health_model.pkl'")
    
    # Проверяем размер файла
    import os
    file_size = os.path.getsize('dog_health_model.pkl')
    print(f"   📦 Размер файла: {file_size / 1024:.2f} KB")
    
except Exception as e:
    print(f"   ❌ Ошибка при сохранении: {e}")
    exit(1)

# ============================================
# 12. ТЕСТОВОЕ ПРЕДСКАЗАНИЕ
# ============================================
print("\n🧪 Шаг 12: Тестовое предсказание...")

# Пример 1: Здоровая кошка
test_cat_healthy = {
    'возраст': 2,
    'вес': 4.5,
    'играет (мин.)': 45,
    'спит (часы)': 14
}

# Пример 2: Кошка с проблемами
test_cat_unhealthy = {
    'возраст': 18,
    'вес': 1.8,
    'играет (мин.)': 5,
    'спит (часы)': 22
}

print("\n   🐱 Тест 1: Здоровая кошка")
print(f"      Возраст: {test_cat_healthy['возраст']} лет")
print(f"      Вес: {test_cat_healthy['вес']} кг")
print(f"      Игры: {test_cat_healthy['играет (мин.)']} мин/день")
print(f"      Сон: {test_cat_healthy['спит (часы)']} час/день")

# Подготовка данных для предсказания
def prepare_test_data(cat_data):
    test_df = pd.DataFrame([cat_data])
    test_df['Age_Group'] = pd.cut(test_df['возраст'], bins=[0, 2, 8, 15, 30], labels=[0, 1, 2, 3])
    test_df['Weight_Group'] = pd.cut(test_df['вес'], bins=[0, 2, 4, 6, 15], labels=[0, 1, 2, 3])
    test_df['Activity_Ratio'] = test_df['играет (мин.)'] / test_df['спит (часы)']
    return test_df[feature_columns]

test_healthy = prepare_test_data(test_cat_healthy)
test_healthy_scaled = scaler.transform(test_healthy)
pred_healthy = model.predict(test_healthy_scaled)[0]
prob_healthy = model.predict_proba(test_healthy_scaled)[0]

result_healthy = le.inverse_transform([pred_healthy])[0]
print(f"\n   📊 Результат: {result_healthy}")
print(f"   📈 Вероятность здоровья: {prob_healthy[1]*100:.1f}%")

print("\n   🐱 Тест 2: Кошка с проблемами")
print(f"      Возраст: {test_cat_unhealthy['возраст']} лет")
print(f"      Вес: {test_cat_unhealthy['вес']} кг")
print(f"      Игры: {test_cat_unhealthy['играет (мин.)']} мин/день")
print(f"      Сон: {test_cat_unhealthy['спит (часы)']} час/день")

test_unhealthy = prepare_test_data(test_cat_unhealthy)
test_unhealthy_scaled = scaler.transform(test_unhealthy)
pred_unhealthy = model.predict(test_unhealthy_scaled)[0]
prob_unhealthy = model.predict_proba(test_unhealthy_scaled)[0]

result_unhealthy = le.inverse_transform([pred_unhealthy])[0]
print(f"\n   📊 Результат: {result_unhealthy}")
print(f"   📈 Вероятность здоровья: {prob_unhealthy[1]*100:.1f}%")

# ============================================
# ЗАВЕРШЕНИЕ
# ============================================
print("\n" + "=" * 60)
print("🎉 ОБУЧЕНИЕ МОДЕЛИ УСПЕШНО ЗАВЕРШЕНО!")
print("=" * 60)
print("\n📁 Файлы созданы:")
print("   ✅ dog_health_model.pkl - обученная модель")
print("\n💡 Дальнейшие действия:")
print("   1. Запустите приложение: streamlit run app.py")
print("   2. Откройте браузер по адресу: http://localhost:8501")
print("=" * 60)
