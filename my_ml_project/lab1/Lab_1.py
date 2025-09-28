import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv("C:/Users/User/PycharmProjects/PythonProject/my_ml_project/world_economics.csv")

nan_matrix = df.isnull()  # Создаем матрицу с True/False
missing_values_count = nan_matrix.sum()  # Считаем количество True (пропущенных значений) в каждом столбце
print(missing_values_count)

#Заполнение пропущенных значений:
try:
    # Заполнение медианой (числовые)
    df['GDP Growth'] = df['GDP Growth'].fillna(df['GDP Growth'].median())
    df['Interest Rate'] = df['Interest Rate'].fillna(df['Interest Rate'].median())
    df['Inflation Rate'] = df['Inflation Rate'].fillna(df['Inflation Rate'].median())
    df['Jobless Rate'] = df['Jobless Rate'].fillna(df['Jobless Rate'].median())

    # Заполнение модой (категориальные)
    df['capital'] = df['capital'].fillna(df['capital'].mode()[0])
    df['borders'] = df['borders'].fillna(df['borders'].mode()[0])

    # Заполнение средним значением (числовые)
    df['Gov. Budget'] = df['Gov. Budget'].fillna(df['Gov. Budget'].mean())
    df['Debt/GDP'] = df['Debt/GDP'].fillna(df['Debt/GDP'].mean())
    df['Current Account'] = df['Current Account'].fillna(df['Current Account'].mean())

except Exception as e:
    print(f"Ошибка: {e}")

print("\nКоличество пропущенных значений после заполнения:")
print(df.isnull().sum())

#Нормализация
scaler = MinMaxScaler()
# Выбор числовых столбцов для нормализации
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("\nДанные после нормализации:")
print(df.head())

#Преобразование в численный формат
df = pd.get_dummies(df, columns = ['capital'], drop_first = True)
df = pd.get_dummies(df, columns = ['borders'], drop_first = True)
print(df.head())
print(df.columns)



