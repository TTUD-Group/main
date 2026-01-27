import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from tienich import get_hanoi_data, create_dist_matrix
from thuattoan import nearest_neighbor, simulated_annealing, held_karp
import argparse
from matplotlib.animation import FuncAnimation

# Parser ở đầu
parser = argparse.ArgumentParser(description="TSP Hà Nội với tùy chọn mưa/giờ cao điểm/steps SA")
parser.add_argument('--rain', action='store_true', default=True, help="Tăng khoảng cách 20% do mưa (default: True)")
parser.add_argument('--no-rain', dest='rain', action='store_false', help="Tắt mưa")
parser.add_argument('--peak', action='store_true', default=True, help="Tăng khoảng cách 50% giờ cao điểm (default: True)")
parser.add_argument('--no-peak', dest='peak', action='store_false', help="Tắt giờ cao điểm")
parser.add_argument('--steps', type=int, default=50000, help="Số steps cho SA (default: 50000)")
args = parser.parse_args()

def main():
    # 1. Chuẩn bị dữ liệu
    df = get_hanoi_data()
    dist_matrix = create_dist_matrix(df, rain=args.rain, peak_hour=args.peak)

    # 2. Chạy thuật toán
    path_nn, cost_nn = nearest_neighbor(dist_matrix)
    path_sa, cost_sa, history_paths = simulated_annealing(dist_matrix, steps=args.steps)
    path_opt, cost_opt = held_karp(dist_matrix)

    if len(history_paths) > 1:  # Chỉ tạo animation nếu có ít nhất 2 frame
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df.lon, df.lat, c='red', s=120, zorder=5, edgecolor='black')
        line, = ax.plot([], [], 'b-o', linewidth=2.5, markersize=8)
        ax.set_title("Quá trình tối ưu Simulated Annealing", fontsize=14)
        ax.set_xlabel("Kinh độ")
        ax.set_ylabel("Vĩ độ")
        ax.grid(True)

        def calculate_cost(path):
            total = 0
            for k in range(len(path) - 1):
                total += dist_matrix[path[k]][path[k + 1]]
            total += dist_matrix[path[-1]][path[0]]
            return total

        def update(frame):
            path = history_paths[frame]
            lons = [df.iloc[k].lon for k in path] + [df.iloc[path[0]].lon]
            lats = [df.iloc[k].lat for k in path] + [df.iloc[path[0]].lat]
            line.set_data(lons, lats)
            current_cost = calculate_cost(path)
            ax.set_title(f"Simulated Annealing - Bước {frame * 500:,} | Cost: {current_cost:.2f} km", fontsize=14)
            return line,

        ani = FuncAnimation(
            fig,
            update,
            frames=len(history_paths),
            interval=1500,
            blit=True,
            repeat=True
        )
        ani.save(
            r'D:\Downloads\sa_animation_slow.gif',
            writer='pillow',
            fps=0.6,
            dpi=120
        )
        print("Đã lưu animation SLOW vào: D:\\Downloads\\sa_animation_slow.gif")
        plt.close(fig)
    else:
        print("Không đủ frame để tạo animation (chỉ có 1 path).")

    print(f"Kết quả Nearest Neighbor: {cost_nn:.2f} km")
    print(f"Kết quả Simulated Annealing: {cost_sa:.2f} km")
    print(f"Kết quả Tối ưu (Held-Karp): {cost_opt:.2f} km")

    # 3. Vẽ biểu đồ với tất cả 3 lộ trình
    plt.figure(figsize=(15, 10))
    plt.scatter(df.lon, df.lat, c='red', s=150, zorder=5, edgecolor='black')

    # Vẽ lộ trình SA (Màu xanh dương, solid)
    lons_sa = [df.iloc[i].lon for i in path_sa] + [df.iloc[path_sa[0]].lon]
    lats_sa = [df.iloc[i].lat for i in path_sa] + [df.iloc[path_sa[0]].lat]
    plt.plot(lons_sa, lats_sa, color='blue', linestyle='-', marker='o', linewidth=2.5, label=f'Lộ trình SA: {cost_sa:.2f}km')

    # Vẽ lộ trình NN (Màu cam, dashed)
    lons_nn = [df.iloc[i].lon for i in path_nn] + [df.iloc[path_nn[0]].lon]
    lats_nn = [df.iloc[i].lat for i in path_nn] + [df.iloc[path_nn[0]].lat]
    plt.plot(lons_nn, lats_nn, color='orange', linestyle='--', marker='o', linewidth=1.5, label=f'Lộ trình NN: {cost_nn:.2f}km')

    # Vẽ lộ trình Optimal (Màu xanh lá, dotted)
    lons_opt = [df.iloc[i].lon for i in path_opt] + [df.iloc[path_opt[0]].lon]
    lats_opt = [df.iloc[i].lat for i in path_opt] + [df.iloc[path_opt[0]].lat]
    plt.plot(lons_opt, lats_opt, color='green', linestyle='-.', marker='o', linewidth=2, label=f'Lộ trình Optimal: {cost_opt:.2f}km')

    for i, name in enumerate(df.name):
        plt.annotate(name, (df.iloc[i].lon + 0.001, df.iloc[i].lat + 0.001), fontsize=10, ha='left')

    plt.title("So sánh và Tối ưu lộ trình giao hàng (TSP) tại Hà Nội (NN, SA, Optimal)")
    plt.xlabel("Kinh độ (Longitude)")
    plt.ylabel("Vĩ độ (Latitude)")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'D:\Downloads\lo_trinh_hanoi_all.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ vào file: lo_trinh_hanoi_all.png")

    # 4. Tạo bản đồ interactive với Folium (thay thế Matplotlib scatter)
    m = folium.Map(location=[21.0285, 105.8521], zoom_start=11, tiles='CartoDB positron')

    # Đánh dấu 10 điểm với popup chi tiết
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"<b>{row['name']}</b><br>Index: {idx}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}",
            tooltip=row['name'],
            icon=folium.Icon(color='red', icon='info-sign', prefix='fa')
        ).add_to(m)

    # Vẽ lộ trình SA (xanh dương, dày, solid)
    points_sa = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_sa] + [[df.iloc[path_sa[0]]['lat'], df.iloc[path_sa[0]]['lon']]]
    folium.PolyLine(
        points_sa,
        color="blue",
        weight=5,
        opacity=0.9,
        tooltip=f"Lộ trình SA: {cost_sa:.2f} km",
        popup=f"Simulated Annealing - Cost: {cost_sa:.2f} km"
    ).add_to(m)

    # Vẽ lộ trình NN (cam, dashed)
    points_nn = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_nn] + [[df.iloc[path_nn[0]]['lat'], df.iloc[path_nn[0]]['lon']]]
    folium.PolyLine(
        points_nn,
        color="orange",
        weight=4,
        opacity=0.8,
        dash_array='10, 5',
        tooltip=f"Lộ trình NN: {cost_nn:.2f} km",
        popup=f"Nearest Neighbor - Cost: {cost_nn:.2f} km"
    ).add_to(m)

    # Vẽ lộ trình Optimal (xanh lá, dotted)
    points_opt = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_opt] + [[df.iloc[path_opt[0]]['lat'], df.iloc[path_opt[0]]['lon']]]
    folium.PolyLine(
        points_opt,
        color="green",
        weight=4,
        opacity=0.85,
        dash_array='3, 6',
        tooltip=f"Lộ trình Optimal: {cost_opt:.2f} km",
        popup=f"Held-Karp Optimal - Cost: {cost_opt:.2f} km"
    ).add_to(m)

    # Lưu bản đồ
    m.save(r'D:\Downloads\lo_trinh_hanoi_interactive.html')
    print("Đã lưu bản đồ tương tác (zoom, click, popup) vào: D:\\Downloads\\lo_trinh_hanoi_interactive.html")
    print("Mở file HTML bằng trình duyệt (Chrome/Firefox) để xem bản đồ Hà Nội thực tế.")

if __name__ == "__main__":
    main()