import pandas as pd
import os

def data_load(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    df = pd.read_csv(filepath)
    return df

def get_basic_info(df: pd.DataFrame) -> None:
    print('Первые 5 столбцов')
    print(df.head())

    print('Типы данных')
    print(df.dtypes)

    print('Пропуски')
    print(df.isnull().sum())

    print('Статистика')
    print(df.describe)


        