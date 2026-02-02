import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from tienich import get_hanoi_data, create_dist_matrix
from thuattoan import nearest_neighbor, simulated_annealing, held_karp
import argparse
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
import os
from datetime import datetime

# Parser ở đầu
parser = argparse.ArgumentParser(description="TSP Hà Nội với tùy chọn mưa/giờ cao điểm/steps SA")
parser.add_argument('--rain', action='store_true', default=True, help="Tăng khoảng cách 20% do mưa (default: True)")
parser.add_argument('--no-rain', dest='rain', action='store_false', help="Tắt mưa")
parser.add_argument('--peak', action='store_true', default=True, help="Tăng khoảng cách 50% giờ cao điểm (default: True)")
parser.add_argument('--no-peak', dest='peak', action='store_false', help="Tắt giờ cao điểm")
parser.add_argument('--steps', type=int, default=50000, help="Số steps cho SA (default: 50000)")
args = parser.parse_args()

# Giả sử tốc độ trung bình (km/h) để tính thời gian di chuyển. Có thể điều chỉnh
SPEED_KMH = 50  # Tốc độ trung bình ở Hà Nội, xem xét giao thông

def calculate_travel_time(distance_km):
    """Tính thời gian di chuyển (giờ) dựa trên khoảng cách và tốc độ."""
    return distance_km / SPEED_KMH

def print_path_details(df, path, dist_matrix, algorithm_name):
    """In chi tiết quãng đường và thời gian cho từng cặp điểm trong path."""
    print(f"\nChi tiết lộ trình {algorithm_name}:")
    total_distance = 0
    total_time_hours = 0
    for i in range(len(path)):
        start_idx = path[i]
        end_idx = path[(i + 1) % len(path)]
        start_name = df.iloc[start_idx]['name']
        end_name = df.iloc[end_idx]['name']
        dist = dist_matrix[start_idx][end_idx]
        time_hours = calculate_travel_time(dist)
        print(f"Từ {start_name} đến {end_name}: {dist:.2f} km, thời gian: {time_hours:.2f} giờ ({time_hours * 60:.0f} phút)")
        total_distance += dist
        total_time_hours += time_hours
    print(f"Tổng quãng đường: {total_distance:.2f} km")
    print(f"Tổng thời gian di chuyển: {total_time_hours:.2f} giờ ({total_time_hours * 60:.0f} phút)")

def main():
    # 1. Chuẩn bị dữ liệu
    df = get_hanoi_data()
    dist_matrix = create_dist_matrix(df, rain=args.rain, peak_hour=args.peak)
    print(f"Điều kiện: Mưa = {args.rain}, Giờ cao điểm = {args.peak}, Steps SA = {args.steps}")

    results = []

    # 2. Chạy và đo thời gian Nearest Neighbor
    start_nn = time.time()
    path_nn, cost_nn = nearest_neighbor(dist_matrix)
    end_nn = time.time()
    time_nn = end_nn - start_nn
    travel_time_nn = calculate_travel_time(cost_nn)
    print(f"Kết quả Nearest Neighbor: {cost_nn:.2f} km, Thời gian chạy: {time_nn:.4f} giây")
    print_path_details(df, path_nn, dist_matrix, "Nearest Neighbor")
    results.append({'Algorithm': 'Nearest Neighbor', 'Cost (km)': cost_nn, 'Runtime (s)': time_nn, 'Travel Time (hours)': travel_time_nn})

    # 3. Chạy và đo thời gian Simulated Annealing
    start_sa = time.time()
    path_sa, cost_sa, history_paths = simulated_annealing(dist_matrix, steps=args.steps)
    end_sa = time.time()
    time_sa = end_sa - start_sa
    travel_time_sa = calculate_travel_time(cost_sa)
    print(f"Kết quả Simulated Annealing: {cost_sa:.2f} km, Thời gian chạy: {time_sa:.4f} giây")
    print_path_details(df, path_sa, dist_matrix, "Simulated Annealing")
    results.append({'Algorithm': 'Simulated Annealing', 'Cost (km)': cost_sa, 'Runtime (s)': time_sa, 'Travel Time (hours)': travel_time_sa})

    # 4. Chạy và đo thời gian Held-Karp
    start_hk = time.time()
    path_hk, cost_hk = held_karp(dist_matrix)
    end_hk = time.time()
    time_hk = end_hk - start_hk
    travel_time_hk = calculate_travel_time(cost_hk)
    print(f"Kết quả Held-Karp: {cost_hk:.2f} km, Thời gian chạy: {time_hk:.4f} giây")
    print_path_details(df, path_hk, dist_matrix, "Held-Karp")
    results.append({'Algorithm': 'Held-Karp', 'Cost (km)': cost_hk, 'Runtime (s)': time_hk, 'Travel Time (hours)': travel_time_hk})

    # 5. Hiển thị bảng so sánh (sử dụng pandas)
    results_df = pd.DataFrame(results)
    print("\nBảng so sánh các thuật toán:")
    print(results_df.to_string(index=False))

    # Thêm các hàm lưu file mới
    def save_comparison_to_csv(results, filename=r'D:\Downloads\ket_qua_tsp_hanoi.csv'):
        """Lưu bảng so sánh các thuật toán vào CSV"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Đã lưu bảng so sánh vào: {filename}")

    def save_route_details_to_txt(df, path_nn, cost_nn, path_sa, cost_sa, path_hk, cost_hk, dist_matrix, filename=r'D:\Downloads\chi_tiet_lo_trinh.txt'):
        """Lưu chi tiết lộ trình đầy đủ của cả 3 thuật toán vào file TXT"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("============================================================\n")
            f.write("          CHI TIẾT LỘ TRÌNH TSP HÀ NỘI\n")
            f.write(f"Ngày chạy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Điều kiện: Mưa = {args.rain}, Giờ cao điểm = {args.peak}\n")
            f.write("============================================================\n\n")

            def write_path_details(path, cost, name):
                f.write(f"--- {name} (Tổng: {cost:.2f} km) ---\n")
                total_time = 0
                for i in range(len(path)):
                    start_idx = path[i]
                    end_idx = path[(i + 1) % len(path)]
                    start_name = df.iloc[start_idx]['name']
                    end_name = df.iloc[end_idx]['name']
                    dist = dist_matrix[start_idx][end_idx]
                    time_hours = calculate_travel_time(dist)
                    f.write(f"{i + 1:2d}. Từ {start_name:<12} → {end_name:<12}: {dist:6.2f} km | {time_hours:5.2f} giờ ({time_hours * 60:5.0f} phút)\n")
                    total_time += time_hours
                f.write(f"Tổng thời gian: {total_time:.2f} giờ ({total_time * 60:.0f} phút)\n")
                f.write("Danh sách thứ tự điểm (index): " + " → ".join(map(str, path)) + f" → {path[0]}\n\n")

            write_path_details(path_nn, cost_nn, "Nearest Neighbor")
            write_path_details(path_sa, cost_sa, "Simulated Annealing")
            write_path_details(path_hk, cost_hk, "Held-Karp (Tối ưu)")

        print(f"Đã lưu chi tiết lộ trình đầy đủ vào: {filename}")

    # 6. Vẽ biểu đồ so sánh (bar chart cho Cost, Runtime, Travel Time)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Biểu đồ Cost
    axs[0].bar(results_df['Algorithm'], results_df['Cost (km)'], color=['orange', 'blue', 'green'])
    axs[0].set_title('So sánh Cost (km)')
    axs[0].set_ylabel('Cost (km)')
    axs[0].tick_params(axis='x', rotation=45)

    # Biểu đồ Runtime
    axs[1].bar(results_df['Algorithm'], results_df['Runtime (s)'], color=['orange', 'blue', 'green'])
    axs[1].set_title('So sánh Thời gian chạy (s)')
    axs[1].set_ylabel('Runtime (s)')
    axs[1].tick_params(axis='x', rotation=45)

    # Biểu đồ Travel Time
    axs[2].bar(results_df['Algorithm'], results_df['Travel Time (hours)'], color=['orange', 'blue', 'green'])
    axs[2].set_title('So sánh Thời gian di chuyển (giờ)')
    axs[2].set_ylabel('Travel Time (hours)')
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(r'D:\Downloads\so_sanh_thuat_toan.png', dpi=300)
    print("Đã lưu biểu đồ so sánh vào: D:\\Downloads\\so_sanh_thuat_toan.png")
    save_comparison_to_csv(results)
    save_route_details_to_txt(df, path_nn, cost_nn, path_sa, cost_sa, path_hk, cost_hk, dist_matrix)

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
    print(f"Kết quả Tối ưu (Held-Karp): {cost_hk:.2f} km")

    # 7. Vẽ biểu đồ với tất cả 3 lộ trình
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
    lons_opt = [df.iloc[i].lon for i in path_hk] + [df.iloc[path_hk[0]].lon]
    lats_opt = [df.iloc[i].lat for i in path_hk] + [df.iloc[path_hk[0]].lat]
    plt.plot(lons_opt, lats_opt, color='green', linestyle='-.', marker='o', linewidth=2, label=f'Lộ trình Optimal: {cost_hk:.2f}km')

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

    # 8. Tạo bản đồ interactive với Folium (thay thế Matplotlib scatter)
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
    points_opt = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_hk] + [[df.iloc[path_hk[0]]['lat'], df.iloc[path_hk[0]]['lon']]]
    folium.PolyLine(
        points_opt,
        color="green",
        weight=4,
        opacity=0.85,
        dash_array='3, 6',
        tooltip=f"Lộ trình Optimal: {cost_hk:.2f} km",
        popup=f"Held-Karp Optimal - Cost: {cost_hk:.2f} km"
    ).add_to(m)

    # Lưu bản đồ
    m.save(r'D:\Downloads\lo_trinh_hanoi_interactive.html')
    print("Đã lưu bản đồ tương tác (zoom, click, popup) vào: D:\\Downloads\\lo_trinh_hanoi_interactive.html")
    print("Mở file HTML bằng trình duyệt (Chrome/Firefox) để xem bản đồ Hà Nội thực tế.")

if __name__ == "__main__":
    main()