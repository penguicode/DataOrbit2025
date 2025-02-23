
import pandas as pd
import random


def generate_safe_coordinates(num_points):
    lat_min, lat_max = 32.436893, 41.717617
    lon_min, lon_max = -124.0000, -114.0001
    
    coordinates = []
    for _ in range(num_points):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        coordinates.append([lat, lon])
    
    return pd.DataFrame(coordinates, columns=['latitude', 'longitude'])


df = pd.read_csv('C:\\Users\\Rex N\\Desktop\\dataorbit\\Data\\Fire\\ca_daily_fire_2000_2021.csv')

unsafe_coordinates_df = df.groupby('year', group_keys=False).apply(lambda x: x.sample(min(len(x), 500), random_state=42))
unsafe_coordinates_df['fire danger'] = 1
unsafe_coordinates_df.to_csv('unsafe_coordinates', index=False)

safe_coordinates_df = generate_safe_coordinates(unsafe_coordinates_df.shape[0])
safe_coordinates_df['fire danger'] = 0

training_data_df = pd.concat([unsafe_coordinates_df[['latitude', 'longitude', 'fire danger']], safe_coordinates_df], ignore_index=True)
training_data_df = training_data_df.sample(frac=1, random_state=42).reset_index(drop=True)
