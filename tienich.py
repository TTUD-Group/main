import pandas as pd
import numpy as np

def get_hanoi_data():
    # Tọa độ 10 địa điểm tại Hà Nội
    data = {
        'name': ['Hoàn Kiếm', 'Ba Đình', 'Tây Hồ', 'Cầu Giấy', 'Đống Đa', 
                 'Hai Bà Trưng', 'Hoàng Mai', 'Thanh Xuân', 'Long Biên', 'Nam Từ Liêm'],
        'lat': [21.0285, 21.0368, 21.0700, 21.0333, 21.0117, 
                21.0125, 20.9667, 20.9937, 21.0400, 21.0100],
        'lon': [105.8521, 105.8342, 105.8200, 105.7833, 105.8250, 
                105.8500, 105.8500, 105.8119, 105.8900, 105.7700]
    }
    return pd.DataFrame(data)

def create_dist_matrix(df, rain=False, peak_hour=False):
    num_points = len(df)
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                # Công thức tính khoảng cách Euclidean quy đổi ra km
                d = np.sqrt((df.iloc[i].lat - df.iloc[j].lat)**2 + 
                            (df.iloc[i].lon - df.iloc[j].lon)**2) * 111
                if rain: d *= 1.2  # Tăng 20% do mưa
                if peak_hour: d *= 1.5  # Tăng 50% giờ cao điểm
                matrix[i][j] = d
    return matrix