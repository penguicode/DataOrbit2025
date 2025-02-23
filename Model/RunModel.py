
import torch
from InitializeModel import *

model = FireRiskModel()

model.load_state_dict(torch.load('fire_risk_model_weights.pth'))

model.eval()

with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f'Test Loss after loading: {test_loss.item():.4f}')
