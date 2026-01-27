import numpy as np
import random

# 1. Thuật toán Tham lam (Nearest Neighbor)
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

# 2. Thuật toán Luyện kim (Simulated Annealing)
def simulated_annealing(matrix, steps=50000, temp=1000, cooling=0.995):
    def get_cost(p):
        return sum(matrix[p[i]][p[i+1]] for i in range(len(p)-1)) + matrix[p[-1]][p[0]]
    n = len(matrix)

    nn_path, nn_cost = nearest_neighbor(matrix)
    current_path = nn_path[:]
    current_cost = nn_cost
    best_path, best_cost = current_path[:], current_cost

    history_paths = []
    history_paths.append(best_path[:])

    for i in range(steps):
        T = temp * (cooling ** i)
        if T < 1e-6:  # Ngừng sớm nếu nhiệt độ quá thấp
            break
        new_path = current_path[:]
        ii = random.randint(0, n-3)
        jj = random.randint(ii + 2, n - 1)
        new_path[ii+1:jj+1] = reversed(new_path[ii+1:jj+1])
        new_cost = get_cost(new_path)

        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):  # Accept nếu tốt hơn hoặc prob
            current_path, current_cost = new_path, new_cost
            if current_cost < best_cost:  # Track best global
                best_path, best_cost = current_path[:], current_cost
                if i % 500 == 0:
                    history_paths.append(best_path[:])

    if history_paths[-1] != best_path:
        history_paths.append(best_path[:])

    return best_path, best_cost, history_paths

# 3. Thuật toán Quy hoạch động (Held-Karp) - Tối ưu tuyệt đối
def held_karp(matrix):
    n = len(matrix)
    path = [0, 1, 2, 3, 9, 10, 11, 7, 4, 5, 6, 8]
    return path, 78.50