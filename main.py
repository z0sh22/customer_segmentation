from src.data_loader import data_load, get_basic_info
from src.preprocessing import select_features, scale
from src.clustering import find_opt, train_kmeans
from src.visualization import plot_elbow, plot_clusters

# Загрузка
df = data_load('data/Mall_Customers.csv')
get_basic_info(df)

# Предобработка
X = select_features(df)
X_scaled, scaler = scale(X)

# Выбор K через Elbow Method
wcss, optimal_k = find_opt(X_scaled)
plot_elbow(wcss, optimal_k)

# Обучение модели
kmeans, labels = train_kmeans(X_scaled, optimal_k)
df['Cluster'] = labels

# Визуализация
plot_clusters(df, kmeans, scaler)

# Итоговая статистика
summary = df.groupby('Cluster').agg(
    Количество=('CustomerID', 'count'),
    Средний_возраст=('Age', 'mean'),
    Средний_доход=('Annual Income (k$)', 'mean'),
    Средние_траты=('Spending Score (1-100)', 'mean')
).round(1)
print(summary)




