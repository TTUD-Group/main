import matplotlib.pyplot as plt
from tienich import get_hanoi_data, create_dist_matrix
from thuattoan import nearest_neighbor, simulated_annealing, held_karp

def main():
    # 1. Chuẩn bị dữ liệu
    df = get_hanoi_data()
    dist_matrix = create_dist_matrix(df, rain=True, peak_hour=True)

    # 2. Chạy thuật toán
    path_nn, cost_nn = nearest_neighbor(dist_matrix)
    path_sa, cost_sa = simulated_annealing(dist_matrix)
    path_opt, cost_opt = held_karp(dist_matrix)

    print(f"Kết quả Nearest Neighbor: {cost_nn:.2f} km")
    print(f"Kết quả Simulated Annealing: {cost_sa:.2f} km")
    print(f"Kết quả Tối ưu (Held-Karp): {cost_opt:.2f} km")

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.scatter(df.lon, df.lat, c='red', s=100, zorder=5)
    
    # Vẽ lộ trình SA (Màu xanh)
    lons = [df.iloc[i].lon for i in path_sa] + [df.iloc[path_sa[0]].lon]
    lats = [df.iloc[i].lat for i in path_sa] + [df.iloc[path_sa[0]].lat]
    plt.plot(lons, lats, 'b-o', linewidth=2, label=f'Lộ trình SA: {cost_sa:.2f}km')

    for i, name in enumerate(df.name):
        plt.annotate(name, (df.iloc[i].lon, df.iloc[i].lat), fontsize=10)

    plt.title("Tối ưu lịch trình giao hàng tại Hà Nội (TSP)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()