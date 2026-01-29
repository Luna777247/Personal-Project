import os
import json
from src.train import build_model, train_small

if __name__ == '__main__':
    os.makedirs('models/smoke_models', exist_ok=True)
    print('Starting smoke training (small subset, 2 epochs)')
    train_small(epochs=2, out_dir='models/smoke_models')
    print('Smoke training complete')