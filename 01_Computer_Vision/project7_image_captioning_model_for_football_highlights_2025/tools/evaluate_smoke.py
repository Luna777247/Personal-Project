import os
import json
from src.evaluate import evaluate_smoke

if __name__ == '__main__':
    os.makedirs('results/smoke_eval', exist_ok=True)
    print('Running smoke evaluation')
    evaluate_smoke(model_path='models/smoke_models/football_caption_model.h5', out_dir='results/smoke_eval')
    print('Smoke evaluation complete')