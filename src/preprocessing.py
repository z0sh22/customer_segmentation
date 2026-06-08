import pandas as pd 
from sklearn.preprocessing import StandardScaler

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].copy()
    return X

def scale(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
