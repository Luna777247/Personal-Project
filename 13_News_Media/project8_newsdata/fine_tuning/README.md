# Fine-tuning Directory for Disaster Information Extraction

This directory contains a comprehensive fine-tuning system for specialized disaster information extraction models optimized for Vietnamese journalism.

## Directory Structure

```
fine_tuning/
├── config/
│   └── config.yaml              # Model configurations and training parameters
├── data/
│   ├── train/                   # Training data
│   ├── val/                     # Validation data
│   └── test/                    # Test data
├── models/                      # Trained models (created after training)
├── scripts/
│   ├── annotate_data.py         # Automatic data annotation
│   ├── train_ner.py            # NER model training
│   ├── train_event_extraction.py # Event extraction training
│   ├── train_relation_extraction.py # Relation extraction training
│   ├── evaluate_models.py      # Model evaluation
│   ├── inference.py            # Production inference
│   └── visualize_results.py    # Results visualization
├── notebooks/                   # Jupyter notebooks for analysis
├── output/                      # Training outputs and logs
└── evaluation/                  # Evaluation results
```

## Features

- **Vietnamese-Optimized Models**: Uses PhoBERT and ViBERT for Vietnamese text processing
- **Multi-Task Learning**: NER, Event Extraction, and Relation Extraction
- **Automatic Annotation**: Rule-based entity extraction for disaster-related information
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Production Ready**: Inference pipeline for real-world deployment

## Entity Types

The system extracts the following disaster-related entities:

- **DISASTER_TYPE**: Type of disaster (flood, earthquake, storm, etc.)
- **LOCATION**: Affected locations and regions
- **TIME**: When the disaster occurred or is occurring
- **DAMAGE**: Damage assessment and statistics
- **RESPONSE**: Government and organizational responses
- **IMPACT**: Human and economic impact
- **FORECAST**: Weather forecasts and predictions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your training data in JSON format in the `data/` directory:

```json
[
  {
    "text": "Hôm nay lũ lụt xảy ra tại Hà Nội gây thiệt hại nặng nề",
    "entities": [
      {
        "start": 8,
        "end": 16,
        "label": "DISASTER_TYPE",
        "text": "lũ lụt"
      },
      {
        "start": 25,
        "end": 31,
        "label": "LOCATION",
        "text": "Hà Nội"
      }
    ],
    "event_type": "FLOOD"
  }
]
```

### 3. Configure Models

Edit `config/config.yaml` to adjust model parameters:

```yaml
model:
  ner_model: "vinai/phobert-base"
  event_model: "vinai/phobert-base"
  relation_model: "vinai/phobert-base"

training:
  ner:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
  event_extraction:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
  relation_extraction:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
```

### 4. Train Models

Train individual models:

```bash
# Train NER model
python scripts/train_ner.py --config config/config.yaml --data data/train/ner_train.json --output models/ner_model

# Train Event Extraction model
python scripts/train_event_extraction.py --config config/config.yaml --data data/train/event_train.json --output models/event_model

# Train Relation Extraction model
python scripts/train_relation_extraction.py --config config/config.yaml --data data/train/relation_train.json --output models/relation_model
```

### 5. Evaluate Models

Evaluate trained models:

```bash
# Evaluate NER model
python scripts/evaluate_models.py --task ner --model-path models/ner_model --test-data data/test/ner_test.json --output-dir evaluation/

# Evaluate all models
python scripts/evaluate_models.py --task all --model-path models/ --test-data data/test/ --output-dir evaluation/
```

### 6. Visualize Results

Create comprehensive visualizations:

```bash
python scripts/visualize_results.py --logs-dir output/ --evaluation-dir evaluation/ --output-dir visualizations/
```

### 7. Run Inference

Extract disaster information from new text:

```bash
# Single text inference
python scripts/inference.py --models-dir models/ --input "Lũ lụt xảy ra tại Nghệ An gây thiệt hại 100 tỷ đồng"

# Batch processing
python scripts/inference.py --models-dir models/ --input news_articles.json --batch --output results.json
```

## Data Format

### NER Data Format

```json
{
  "text": "Full news article text",
  "entities": [
    {
      "start": 10,
      "end": 20,
      "label": "DISASTER_TYPE",
      "text": "lũ lụt"
    }
  ]
}
```

### Event Extraction Data Format

```json
{
  "text": "Full news article text",
  "event_type": "FLOOD"
}
```

### Relation Extraction Data Format

```json
{
  "text": "Full news article text",
  "entities": [...],
  "relations": [
    {
      "head": {"start": 10, "end": 20, "text": "lũ lụt", "label": "DISASTER_TYPE"},
      "tail": {"start": 30, "end": 40, "text": "Hà Nội", "label": "LOCATION"},
      "relation_type": "OCCURS_IN"
    }
  ]
}
```

## Model Architectures

### NER Model
- **Base Model**: PhoBERT/ViBERT
- **Task**: Token Classification
- **Labels**: B-/I- tags for each entity type

### Event Extraction Model
- **Base Model**: PhoBERT/ViBERT
- **Task**: Sequence Classification
- **Labels**: Disaster event types (FLOOD, EARTHQUAKE, STORM, etc.)

### Relation Extraction Model
- **Base Model**: PhoBERT/ViBERT
- **Task**: Sequence Classification with entity markers
- **Labels**: Relation types (OCCURS_IN, CAUSES, RESPONDS_TO, etc.)

## Configuration Options

### Model Configuration

```yaml
model:
  # Pretrained models
  ner_model: "vinai/phobert-base"
  event_model: "vinai/phobert-base"
  relation_model: "vinai/phobert-base"

  # Model parameters
  max_length: 512
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
```

### Training Configuration

```yaml
training:
  ner:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
    weight_decay: 0.01
    warmup_steps: 500
    save_steps: 500
    eval_steps: 500

  event_extraction:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
    weight_decay: 0.01
    warmup_steps: 500

  relation_extraction:
    epochs: 10
    batch_size: 16
    learning_rate: 5e-5
    weight_decay: 0.01
    warmup_steps: 500
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Macro F1**: Average F1 across all classes
- **Weighted F1**: F1 weighted by class support

## Visualization Outputs

The visualization script generates:

- Training curves (loss, F1, accuracy, learning rate)
- Model comparison charts
- Confusion matrices
- Per-class performance analysis
- Training time analysis
- Comprehensive HTML report

## Production Deployment

For production use:

1. Train models with optimal hyperparameters
2. Evaluate on held-out test set
3. Use `inference.py` for batch processing
4. Monitor model performance over time
5. Retrain periodically with new data

## Integration with Existing System

This fine-tuning system is designed to complement the existing RAG-based disaster extraction:

1. Use fine-tuned models for initial entity extraction
2. Feed extracted entities into RAG system for context enrichment
3. Combine structured extraction with semantic search

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in config
2. **Poor Performance**: Check data quality and increase training epochs
3. **Convergence Issues**: Adjust learning rate or use different optimizer
4. **Label Imbalance**: Use class weights or oversampling

### Performance Optimization

1. Use gradient checkpointing for large models
2. Implement mixed precision training
3. Use data parallelism for multi-GPU training
4. Optimize data loading with proper batching

## Contributing

1. Follow the existing code style and structure
2. Add comprehensive docstrings and comments
3. Include unit tests for new functionality
4. Update documentation for any changes

## License

This project is part of the disaster information extraction system for Vietnamese journalism.