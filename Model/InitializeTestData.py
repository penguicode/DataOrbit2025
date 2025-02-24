
import numpy as np
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


def generate_safe_humidity(month_series):
    humidity_per_month = {
        1: .71, 2: .69, 3: .67, 4: .62, 5: .61, 6: .66,
        7: .61, 8: .63, 9: .62, 10: .64, 11: .68, 12: .71 
    }

    return month_series.map(humidity_per_month)


def generate_safe_month(num_points):
    months = [random.randrange(1, 13) for _ in range(num_points)]
    return pd.DataFrame(months, columns=['month'])

def generate_safe_fire_danger(num_points):
    fire_danger = [round(random.random() ** 2, 2) for _ in range(num_points)]
    return pd.DataFrame(fire_danger, columns=['fire danger'])


df = pd.read_csv('C:\\Users\\Rex N\\Desktop\\DataOrbit2025\\Model\\unsafe_coordinates')

unsafe_data_df = df.groupby('year', group_keys=False).apply(lambda x: x.sample(min(len(x), 250), random_state=42))
unsafe_data_df['fire danger'] = 1.0
unsafe_data_df.to_csv('unsafe_coordinates', index=False)
unsafe_data_df['humidity'] = generate_safe_humidity(unsafe_data_df['month'])

safe_data_df = generate_safe_coordinates(unsafe_data_df.shape[0])
safe_data_df['fire danger'] = generate_safe_fire_danger(unsafe_data_df.shape[0])['fire danger']
safe_data_df['month'] = generate_safe_month(unsafe_data_df.shape[0])
safe_data_df['humidity'] = generate_safe_humidity(safe_data_df['month'])

training_data_df = pd.concat([
    unsafe_data_df[['latitude', 'longitude', 'month', 'fire danger', 'humidity']], 
    safe_data_df[['latitude', 'longitude', 'month', 'fire danger', 'humidity']]
    ], ignore_index=True)

training_data_df['month sin'] = np.sin(2 * np.pi * training_data_df['month'] / 12)
training_data_df['month cos'] = np.cos(2 * np.pi * training_data_df['month'] / 12)

training_data_df = training_data_df.drop(['month'], axis=1)

training_data_df = training_data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(training_data_df[:10])