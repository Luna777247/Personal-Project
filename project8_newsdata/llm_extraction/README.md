# LLM-Based Disaster Information Extraction System

## Overview

This system provides advanced disaster information extraction from Vietnamese news articles using Large Language Models (LLMs). It supports multiple LLM providers (OpenAI, Anthropic, Groq) with automatic fallback, cost management, and Vietnamese-optimized prompts.

## Features

- **Multi-Provider Support**: OpenAI GPT, Anthropic Claude, Groq Llama models
- **Vietnamese Optimization**: Specialized prompts for Vietnamese disaster text
- **Cost Management**: Budget tracking, automatic fallback to cheaper models
- **Caching**: Intelligent caching to reduce API costs and improve performance
- **Batch Processing**: Efficient processing of multiple articles
- **Confidence Scoring**: Quality assessment of extractions
- **Error Handling**: Robust error handling with automatic retries
- **Metrics Tracking**: Comprehensive performance and cost metrics

## Installation

### Prerequisites

- Python 3.8+
- API keys for at least one LLM provider:
  - `OPENAI_API_KEY` for GPT models
  - `ANTHROPIC_API_KEY` for Claude models
  - `GROQ_API_KEY` for Llama models

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GROQ_API_KEY="your-groq-key"
   ```

3. **Run the demo:**
   ```bash
   python scripts/demo_llm_extraction.py
   ```

## Usage

### Command Line Interface

The `run.py` script provides a comprehensive CLI for extraction:

```bash
# Extract from text
python run.py extract --text "Bão số 12 gây thiệt hại tại Quảng Nam..."

# Extract from file
python run.py extract --input disaster_news.json --output results.json

# Extract from CSV
python run.py extract --input news.csv --output results.csv --format csv

# Show available models
python run.py models

# Show metrics
python run.py metrics
```

### Python API

```python
from scripts.llm_extractor import LLMExtractor

# Initialize extractor
extractor = LLMExtractor()

# Extract from single text
result = extractor.extract_disaster_info("Bão số 12 gây thiệt hại...")
print(result.extracted_info)

# Extract from multiple texts
results = extractor.extract_from_texts(["text1", "text2"])
for result in results:
    print(result.extracted_info)

# Get metrics
metrics = extractor.get_metrics()
print(f"Success rate: {metrics['success_rate']:.1%}")
```

## Configuration

### Model Configuration

Models are configured in `config/llm_config.py`:

```python
LLM_CONFIGS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "cost_per_1k_tokens": 0.0015,
        "max_tokens": 4096,
        "temperature": 0.1
    },
    # ... more models
}
```

### Prompt Configuration

Prompts are defined in `config/prompts.py` with Vietnamese optimization:

- `basic`: Simple extraction
- `detailed`: Comprehensive extraction
- `full`: Maximum detail with confidence scoring

## Output Format

### JSON Output Structure

```json
{
  "metadata": {
    "total_extractions": 1,
    "timestamp": "2024-01-15T10:30:00Z",
    "format_version": "1.0"
  },
  "results": [
    {
      "id": 1,
      "model": "gpt-3.5-turbo",
      "processing_time": 2.34,
      "cost_estimate": 0.0045,
      "confidence_score": 0.85,
      "extracted_info": {
        "type": "Bão",
        "location": "Quảng Nam",
        "time": "15/11/2023",
        "severity": "Nặng",
        "damage": "1.200 tỷ đồng",
        "deaths": "15",
        "injured": "27",
        "missing": "5",
        "organizations": ["Đội cứu hộ", "Ủy ban nhân dân"],
        "forecast": "Mưa lớn tiếp tục"
      },
      "text_preview": "Bão số 12 gây thiệt hại nặng nề..."
    }
  ]
}
```

### CSV Output Format

| id | model | processing_time | cost_estimate | confidence_score | type | location | time | severity | damage | deaths | injured | missing | organizations | forecast | text_preview |
|----|-------|-----------------|---------------|------------------|------|----------|------|----------|--------|--------|---------|---------|---------------|----------|--------------|
| 1  | gpt-3.5-turbo | 2.34 | 0.0045 | 0.85 | Bão | Quảng Nam | 15/11/2023 | Nặng | 1.200 tỷ đồng | 15 | 27 | 5 | Đội cứu hộ;Ủy ban nhân dân | Mưa lớn tiếp tục | Bão số 12... |

## Supported Disaster Types

The system can extract information about:

- **Natural Disasters**: Bão (Storm), Lũ (Flood), Động đất (Earthquake), Hạn hán (Drought), Sạt lở (Landslide)
- **Location**: Provinces, districts, specific areas in Vietnam
- **Time**: Dates, times, durations
- **Impact**: Deaths, injuries, missing persons, property damage
- **Response**: Organizations involved, rescue operations
- **Forecast**: Future predictions, warnings

## Performance & Cost Optimization

### Caching Strategy

- Automatic caching of identical texts
- TTL-based cache expiration
- Memory-efficient storage

### Cost Management

- Budget limits per session
- Automatic fallback to cheaper models
- Cost tracking and reporting
- Rate limiting to prevent overuse

### Performance Features

- Async processing for concurrent requests
- Batch processing for multiple texts
- Intelligent model selection based on cost/performance
- Request retry with exponential backoff

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Automatic retry with different models
- **Rate Limits**: Intelligent backoff and queuing
- **Invalid Responses**: Response validation and reprocessing
- **Network Issues**: Timeout handling and reconnection
- **Cost Limits**: Budget exceeded notifications

## Metrics & Monitoring

Track extraction performance:

```python
metrics = extractor.get_metrics()
# {
#   'total_requests': 150,
#   'success_rate': 0.92,
#   'total_cost': 0.234,
#   'avg_processing_time': 1.8,
#   'cache_hit_rate': 0.45,
#   'available_models': ['gpt-3.5-turbo', 'claude-3-haiku']
# }
```

## Integration

### With Existing Systems

The LLM extraction system integrates with existing extraction methods:

```python
# Combine with NER and pattern extraction
from ner_extraction.scripts.ner_extractor import NERExtractor
from pattern_extraction.scripts.pattern_extractor import PatternExtractor

# Multi-stage extraction pipeline
def comprehensive_extraction(text):
    # Stage 1: LLM extraction (contextual understanding)
    llm_result = llm_extractor.extract_disaster_info(text)

    # Stage 2: NER extraction (entity recognition)
    ner_result = ner_extractor.extract_entities(text)

    # Stage 3: Pattern extraction (rule-based)
    pattern_result = pattern_extractor.extract_patterns(text)

    # Combine results
    return combine_results(llm_result, ner_result, pattern_result)
```

### Data Pipeline Integration

```python
# Process news data pipeline
def process_news_pipeline(news_data):
    results = []

    for article in news_data:
        result = extractor.extract_disaster_info(article['content'])
        results.append({
            'article_id': article['id'],
            'extraction': result.extracted_info,
            'confidence': result.confidence_score,
            'cost': result.cost_estimate
        })

    return results
```

## Troubleshooting

### Common Issues

1. **No API Keys**: Set at least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY)

2. **Rate Limits**: The system automatically handles rate limits with backoff

3. **Cost Exceeded**: Monitor budget limits and switch to cheaper models

4. **Poor Quality**: Try different models or prompt types

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Basic Extraction

```python
extractor = LLMExtractor()
text = "Bão Noru gây thiệt hại 80 tỷ đồng tại Phú Yên"
result = extractor.extract_disaster_info(text)

print(result.extracted_info)
# {
#   'type': 'Bão',
#   'location': 'Phú Yên',
#   'damage': '80 tỷ đồng',
#   'deaths': None,
#   ...
# }
```

### Batch Processing

```python
texts = ["Article 1...", "Article 2...", "Article 3..."]
results = extractor.extract_from_texts(texts, batch_size=3)

for result in results:
    print(f"Cost: ${result.cost_estimate:.4f}")
    print(f"Info: {result.extracted_info}")
```

### Custom Configuration

```python
# Use specific model with custom settings
result = extractor.extract_disaster_info(
    text,
    model="gpt-4",
    prompt_type="full",
    temperature=0.0
)
```

## Contributing

1. Test with Vietnamese disaster news
2. Validate extraction quality
3. Monitor costs and performance
4. Report issues and improvements

## License

This project is part of the comprehensive disaster information extraction system.