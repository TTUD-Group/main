import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import folium
from streamlit_folium import folium_static
import time
from io import BytesIO
from itertools import combinations
import os
import json  # Thêm để đọc GeoJSON


# Copy hàm từ tienich.py (updated với GeoJSON)
def get_hanoi_data():
    # Đọc GeoJSON từ file (toàn VN POI, không lọc Hà Nội)
    geojson_file = 'hotosm_vnm_points_of_interest_points_geojson.geojson'  # Tên file unzip
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract name, lat, lon
    points = []
    for feature in data['features']:
        if 'properties' in feature and 'geometry' in feature and feature['geometry']['type'] == 'Point':
            name = feature['properties'].get('name', f"POI {len(points) + 1}")  # Default nếu thiếu name
            lon, lat = feature['geometry']['coordinates']  # GeoJSON là [lon, lat]
            points.append({'name': name, 'lat': lat, 'lon': lon})

    df = pd.DataFrame(points)

    # Giới hạn 300 điểm để TSP khả thi (toàn dataset quá lớn)
    num_points = 300
    df = df.head(num_points)  # Chỉnh lên 200 nếu muốn, nhưng SA chậm hơn

    return df


def create_dist_matrix(df, rain=False, peak_hour=False):
    num_points = len(df)
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                d = np.sqrt((df.iloc[i].lat - df.iloc[j].lat) ** 2 +
                            (df.iloc[i].lon - df.iloc[j].lon) ** 2) * 111
                if rain: d *= 1.2
                if peak_hour: d *= 1.5
                matrix[i][j] = d
    return matrix


# Copy hàm từ thuattoan.py (đã cập nhật SA)
def nearest_neighbor(matrix):
    n = len(matrix)
    visited = [False] * n
    path = [0]
    visited[0] = True
    total_cost = 0
    for _ in range(n - 1):
        last = path[-1]
        next_node = np.argmin([matrix[last][j] if not visited[j] else np.inf for j in range(n)])
        path.append(next_node)
        visited[next_node] = True
        total_cost += matrix[last][next_node]
    total_cost += matrix[path[-1]][path[0]]
    return path, total_cost


def simulated_annealing(matrix, steps=100000, temp=1000, cooling=0.999):
    def get_cost(p):
        return sum(matrix[p[i]][p[i + 1]] for i in range(len(p) - 1)) + matrix[p[-1]][p[0]]

    n = len(matrix)
    # Bắt đầu từ random path để khác biệt lớn hơn
    random_path = list(range(n))
    np.random.shuffle(random_path)
    current_path = random_path
    current_cost = get_cost(current_path)
    best_path, best_cost = current_path[:], current_cost
    history_paths = []
    history_paths.append(best_path[:])
    for i in range(steps):
        T = temp * (cooling ** i)
        if T < 1e-6:
            break
        new_path = current_path[:]
        ii = np.random.randint(0, n - 3)
        jj = np.random.randint(ii + 2, n - 1)
        new_path[ii + 1:jj + 1] = reversed(new_path[ii + 1:jj + 1])
        new_cost = get_cost(new_path)
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_path, current_cost = new_path, new_cost
            if current_cost < best_cost:
                best_path, best_cost = current_path[:], current_cost
                if i % 100 == 0:  # Track thường xuyên hơn
                    history_paths.append(best_path[:])
    if history_paths[-1] != best_path:
        history_paths.append(best_path[:])
    return best_path, best_cost, history_paths


def held_karp(dists):
    n = len(dists)
    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    if (prev, m) in C:
                        res.append((C[(prev, m)][0] + dists[m][k], m))
                if res:
                    C[(bits, k)] = min(res)
    bits = (2 ** n - 1) - 1
    res = []
    for k in range(1, n):
        if (bits, k) in C:
            res.append((C[(bits, k)][0] + dists[k][0], k))
    if not res:
        raise ValueError("No path found")
    opt, parent = min(res)
    path = []
    current_bits = bits
    for _ in range(n - 1):
        path.append(parent)
        if (current_bits, parent) in C:
            _, next_parent = C[(current_bits, parent)]
            current_bits &= ~(1 << parent)
            parent = next_parent
        else:
            raise ValueError("Path reconstruction failed")
    path.append(0)
    path.reverse()
    return path, opt


# Các hàm hỗ trợ
SPEED_KMH = 50


def calculate_travel_time(distance_km):
    return distance_km / SPEED_KMH


def get_path_details(df, path, dist_matrix):
    details = []
    total_distance = 0
    total_time_hours = 0
    for i in range(len(path)):
        start_idx = path[i]
        end_idx = path[(i + 1) % len(path)]
        start_name = df.iloc[start_idx]['name']
        end_name = df.iloc[end_idx]['name']
        dist = dist_matrix[start_idx][end_idx]
        time_hours = calculate_travel_time(dist)
        details.append(f"Từ {start_name} đến {end_name}: {dist:.2f} km, thời gian: {time_hours:.2f} giờ ({time_hours * 60:.0f} phút)")
        total_distance += dist
        total_time_hours += time_hours
    details.append(f"Tổng quãng đường: {total_distance:.2f} km")
    details.append(f"Tổng thời gian di chuyển: {total_time_hours:.2f} giờ ({total_time_hours * 60:.0f} phút)")
    return '\n'.join(details)


# Giao diện Streamlit
st.title("TSP Việt Nam POI - Tối ưu Lộ trình Giao hàng")

st.sidebar.header("Tham số đầu vào")
rain = st.sidebar.checkbox("Có mưa (tăng 20% khoảng cách)", value=True)
peak_hour = st.sidebar.checkbox("Giờ cao điểm (tăng 50% khoảng cách)", value=True)
steps = st.sidebar.slider("Số steps cho Simulated Annealing", min_value=1000, max_value=200000, value=100000, step=1000)  # Tăng default

if st.sidebar.button("Chạy TSP"):
    with st.spinner("Đang tính toán... (có thể mất vài giây với SA)"):
        df = get_hanoi_data()
        dist_matrix = create_dist_matrix(df, rain=rain, peak_hour=peak_hour)

        results = []

        # Nearest Neighbor
        start_nn = time.time()
        path_nn, cost_nn = nearest_neighbor(dist_matrix)
        time_nn = time.time() - start_nn
        travel_time_nn = calculate_travel_time(cost_nn)
        results.append({'Algorithm': 'Nearest Neighbor', 'Cost (km)': cost_nn, 'Runtime (s)': time_nn, 'Travel Time (hours)': travel_time_nn})

        # Simulated Annealing
        start_sa = time.time()
        path_sa, cost_sa, history_paths = simulated_annealing(dist_matrix, steps=steps)
        time_sa = time.time() - start_sa
        travel_time_sa = calculate_travel_time(cost_sa)
        results.append({'Algorithm': 'Simulated Annealing', 'Cost (km)': cost_sa, 'Runtime (s)': time_sa, 'Travel Time (hours)': travel_time_sa})

        # Held-Karp (skip nếu n > 20)
        if len(df) <= 20:
            start_hk = time.time()
            path_hk, cost_hk = held_karp(dist_matrix)
            time_hk = time.time() - start_hk
            travel_time_hk = calculate_travel_time(cost_hk)
            results.append({'Algorithm': 'Held-Karp', 'Cost (km)': cost_hk, 'Runtime (s)': time_hk, 'Travel Time (hours)': travel_time_hk})
        else:
            st.warning("Bỏ qua Held-Karp vì số lượng điểm quá lớn (>20), thời gian tính toán tăng theo cấp số nhân.")
            results.append({'Algorithm': 'Held-Karp (Skipped)', 'Cost (km)': 'N/A', 'Runtime (s)': 'N/A', 'Travel Time (hours)': 'N/A'})
            path_hk, cost_hk = path_sa, cost_sa  # Placeholder cho vẽ

        results_df = pd.DataFrame(results)

    st.subheader("Bảng so sánh các thuật toán")
    st.dataframe(results_df)

    st.subheader("Chi tiết lộ trình Nearest Neighbor")
    st.text(get_path_details(df, path_nn, dist_matrix))

    st.subheader("Chi tiết lộ trình Simulated Annealing")
    st.text(get_path_details(df, path_sa, dist_matrix))

    if len(df) <= 20:
        st.subheader("Chi tiết lộ trình Held-Karp")
        st.text(get_path_details(df, path_hk, dist_matrix))
    else:
        st.subheader("Chi tiết lộ trình Held-Karp")
        st.text("N/A - Skipped do n > 20")

    # Biểu đồ so sánh
    plot_df = results_df[results_df['Algorithm'] != 'Held-Karp (Skipped)'].copy()

    if not plot_df.empty:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].bar(plot_df['Algorithm'], plot_df['Cost (km)'], color=['orange', 'blue'])
        axs[0].set_title('So sánh Cost (km)')
        axs[0].set_ylabel('Cost (km)')
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].bar(plot_df['Algorithm'], plot_df['Runtime (s)'], color=['orange', 'blue'])
        axs[1].set_title('So sánh Thời gian chạy (s)')
        axs[1].set_ylabel('Runtime (s)')
        axs[1].tick_params(axis='x', rotation=45)

        axs[2].bar(plot_df['Algorithm'], plot_df['Travel Time (hours)'], color=['orange', 'blue'])
        axs[2].set_title('So sánh Thời gian di chuyển (giờ)')
        axs[2].set_ylabel('Travel Time (hours)')
        axs[2].tick_params(axis='x', rotation=45)

        st.pyplot(fig)
    else:
        st.info("Không có dữ liệu số để vẽ biểu đồ so sánh (Held-Karp bị skip).")

    # Animation Simulated Annealing
    if len(history_paths) > 1:
        fig_anim, ax = plt.subplots(figsize=(10, 7))
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
            ax.set_title(f"Simulated Annealing - Bước {frame * 100:,} | Cost: {current_cost:.2f} km", fontsize=14)
            return line,


        ani = FuncAnimation(fig_anim, update, frames=len(history_paths), interval=1500, blit=True, repeat=True)

        temp_gif_path = "temp_sa_animation.gif"
        try:
            ani.save(temp_gif_path, writer='pillow', fps=0.6, dpi=120)
            with open(temp_gif_path, "rb") as f:
                gif_data = f.read()
            buf = BytesIO(gif_data)
            buf.seek(0)
            st.subheader("Animation Quá trình Simulated Annealing")
            st.image(buf, use_column_width=True)
        except Exception as e:
            st.error(f"Lỗi khi tạo animation: {e}")
        finally:
            if os.path.exists(temp_gif_path):
                os.remove(temp_gif_path)
        plt.close(fig_anim)
    else:
        st.info("Không đủ frame để tạo animation Simulated Annealing.")

    # Biểu đồ lộ trình
    fig_routes = plt.figure(figsize=(15, 10))
    plt.scatter(df.lon, df.lat, c='red', s=150, zorder=5, edgecolor='black')
    lons_sa = [df.iloc[i].lon for i in path_sa] + [df.iloc[path_sa[0]].lon]
    lats_sa = [df.iloc[i].lat for i in path_sa] + [df.iloc[path_sa[0]].lat]
    plt.plot(lons_sa, lats_sa, color='blue', linestyle='-', marker='o', linewidth=2.5, label=f'Lộ trình SA: {cost_sa:.2f}km')
    lons_nn = [df.iloc[i].lon for i in path_nn] + [df.iloc[path_nn[0]].lon]
    lats_nn = [df.iloc[i].lat for i in path_nn] + [df.iloc[path_nn[0]].lat]
    plt.plot(lons_nn, lats_nn, color='orange', linestyle='--', marker='o', linewidth=1.5, label=f'Lộ trình NN: {cost_nn:.2f}km')
    if len(df) <= 20:
        lons_hk = [df.iloc[i].lon for i in path_hk] + [df.iloc[path_hk[0]].lon]
        lats_hk = [df.iloc[i].lat for i in path_hk] + [df.iloc[path_hk[0]].lat]
        plt.plot(lons_hk, lats_hk, color='green', linestyle='-.', marker='o', linewidth=2, label=f'Lộ trình Optimal: {cost_hk:.2f}km')
    for i, name in enumerate(df.name):
        plt.annotate(name, (df.iloc[i].lon + 0.001, df.iloc[i].lat + 0.001), fontsize=10, ha='left')
    plt.title("So sánh Lộ trình TSP tại Việt Nam POI (NN, SA, Optimal)")
    plt.xlabel("Kinh độ (Longitude)")
    plt.ylabel("Vĩ độ (Latitude)")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    st.subheader("Biểu đồ So sánh Lộ trình")
    st.pyplot(fig_routes)

    # Bản đồ Folium tương tác (center VN)
    m = folium.Map(location=[16.0471, 108.2062], zoom_start=5, tiles='CartoDB positron')
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"<b>{row['name']}</b><br>Index: {idx}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}",
            tooltip=row['name'],
            icon=folium.Icon(color='red', icon='info-sign', prefix='fa')
        ).add_to(m)
    points_sa = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_sa] + [[df.iloc[path_sa[0]]['lat'], df.iloc[path_sa[0]]['lon']]]
    folium.PolyLine(points_sa, color="blue", weight=5, opacity=0.9, tooltip=f"Lộ trình SA: {cost_sa:.2f} km", popup=f"Simulated Annealing - Cost: {cost_sa:.2f} km").add_to(m)
    points_nn = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_nn] + [[df.iloc[path_nn[0]]['lat'], df.iloc[path_nn[0]]['lon']]]
    folium.PolyLine(points_nn, color="orange", weight=4, opacity=0.8, dash_array='10, 5', tooltip=f"Lộ trình NN: {cost_nn:.2f} km", popup=f"Nearest Neighbor - Cost: {cost_nn:.2f} km").add_to(m)
    if len(df) <= 20:
        points_hk = [[df.iloc[i]['lat'], df.iloc[i]['lon']] for i in path_hk] + [[df.iloc[path_hk[0]]['lat'], df.iloc[path_hk[0]]['lon']]]
        folium.PolyLine(points_hk, color="green", weight=4, opacity=0.85, dash_array='3, 6', tooltip=f"Lộ trình Optimal: {cost_hk:.2f} km", popup=f"Held-Karp Optimal - Cost: {cost_hk:.2f} km").add_to(m)
    st.subheader("Bản đồ Tương tác (Zoom, Click để xem chi tiết)")
    folium_static(m)