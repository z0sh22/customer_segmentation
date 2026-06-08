import matplotlib.pyplot as plt
import numpy as np


def plot_elbow(wcss: list, optimal_k: int) -> None:
    """График метода локтя с отметкой оптимального K."""
    k_range = range(1, len(wcss) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, wcss, marker='o', color='steelblue', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                label=f'Оптимальное K = {optimal_k}')
    plt.title('Elbow Method — выбор оптимального K')
    plt.xlabel('Количество кластеров (K)')
    plt.ylabel('WCSS')
    plt.xticks(k_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/elbow_method.png', dpi=150)
    plt.show()
    print("✅ График сохранён: reports/elbow_method.png")


def plot_clusters(df, kmeans, scaler) -> None:
    """Визуализация кластеров с центроидами."""
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    n_clusters = kmeans.n_clusters

    plt.figure(figsize=(10, 7))

    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        plt.scatter(
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            c=colors[i], label=f'Кластер {i}',
            s=80, alpha=0.8, edgecolors='white', linewidth=0.5
        )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', marker='X', s=200,
                label='Центроиды', zorder=5)

    plt.title('Сегментация клиентов торгового центра', fontsize=14)
    plt.xlabel('Годовой доход (k$)', fontsize=12)
    plt.ylabel('Оценка трат (1-100)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/clusters.png', dpi=150)
    plt.show()
    print("✅ График сохранён: reports/clusters.png")