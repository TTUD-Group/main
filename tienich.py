import pandas as pd
import numpy as np
import json

def get_hanoi_data():
    geojson_file = 'hotosm_vnm_points_of_interest_points_geojson.geojson'  # Tên file unzip
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract name, lat, lon
    points = []
    for feature in data['features']:
        if 'properties' in feature and 'geometry' in feature and feature['geometry']['type'] == 'Point':
            props = feature['properties'] or {}  # tránh None
            name = props.get('name')
            if name is None or not isinstance(name, str) or name.strip() == "":
                name = f"POI {len(points) + 1}"
            lon, lat = feature['geometry']['coordinates']
            points.append({'name': name, 'lat': lat, 'lon': lon})

    df = pd.DataFrame(points)
    df = df.head(300)

    df['name'] = df['name'].fillna("Unknown").astype(str)

    return df

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