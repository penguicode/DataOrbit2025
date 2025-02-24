
import torch
import pandas as pd
import numpy as np
from InitializeModel import *

model = FireRiskModel()

model.load_state_dict(torch.load('fire_risk_model_weights.pth'))

model.eval()

df = pd.read_csv('download.csv')

test_data = df[['latitude', 'longitude', 'month', 'humidity']].values

test_data_scaled = scaler.transform(test_data)

test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)

model.eval()

with torch.no_grad():
    prediction = model(test_data_tensor)

print(f"Prediction (Fire Risk): {prediction.item():.4f}")
