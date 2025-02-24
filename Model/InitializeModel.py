
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from InitializeTestData import training_data_df
import pandas as pd
import numpy as np


class FireRiskModel(nn.Module):
    def __init__(self):
        super(FireRiskModel, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


X = training_data_df[['latitude', 'longitude', 'month sin', 'month cos', 'humidity']].values
y = training_data_df['fire danger'].values.reshape(-1, 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

model = FireRiskModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0075)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()

    # if epoch % 100 == 0:
    #     print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'fire_risk_model_weights.pth')
