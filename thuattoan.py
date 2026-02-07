import numpy as np
import random
from itertools import combinations

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

# 2. Thuật toán Luyện kim (Simulated Annealing) - Cập nhật để khác biệt hơn
def simulated_annealing(matrix, steps=100000, temp=1000, cooling=0.999):  # Tăng steps, cooling chậm hơn
    def get_cost(p):
        return sum(matrix[p[i]][p[i+1]] for i in range(len(p)-1)) + matrix[p[-1]][p[0]]
    n = len(matrix)

    # Bắt đầu từ random path thay vì NN để khác biệt lớn hơn
    random_path = list(range(n))
    random.shuffle(random_path)
    current_path = random_path
    current_cost = get_cost(current_path)
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
                if i % 100 == 0:  # Track thường xuyên hơn để animation chi tiết
                    history_paths.append(best_path[:])

    if history_paths[-1] != best_path:
        history_paths.append(best_path[:])

    return best_path, best_cost, history_paths

# 3. Thuật toán Quy hoạch động (Held-Karp) - Tối ưu tuyệt đối
def held_karp(dists):
    n = len(dists)
    C = {}
    # Base cases: cost from 0 to each single node
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)
    # Build DP table
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
    # Find min cost for full set
    bits = (2**n - 1) - 1  # all nodes except 0
    res = []
    for k in range(1, n):
        if (bits, k) in C:
            res.append((C[(bits, k)][0] + dists[k][0], k))
    if not res:
        raise ValueError("No path found")
    opt, parent = min(res)
    # Reconstruct path
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