# Evaluation Methodology - Banking Chatbot

## Overview

This document describes the evaluation metrics and methodology used to assess the quality of RAG responses in the Banking Chatbot system.

## Evaluation Metrics

### 1. ROUGE Scores

**Purpose**: Measure n-gram overlap between generated and reference text

**Metrics**:
- **ROUGE-1**: Unigram overlap (word-level matching)
- **ROUGE-2**: Bigram overlap (phrase-level matching)
- **ROUGE-L**: Longest common subsequence

**Scores Reported**:
- Precision: Correct tokens / Generated tokens
- Recall: Correct tokens / Reference tokens
- F1: Harmonic mean of precision and recall

**Interpretation**:
- **0.0-0.3**: Poor overlap
- **0.3-0.5**: Moderate overlap
- **0.5-0.7**: Good overlap
- **0.7-1.0**: Excellent overlap

**Usage**:
```python
from src.evaluation import ROUGEMetric

rouge = ROUGEMetric()
scores = rouge.compute(
    prediction="Lãi suất tiết kiệm là 6.5%/năm",
    reference="Lãi suất tiết kiệm MB Bank là 6.5% mỗi năm"
)
```

### 2. BERTScore

**Purpose**: Semantic similarity using contextual embeddings

**Model**: `vinai/phobert-base` (Vietnamese BERT)

**How It Works**:
1. Generate embeddings for each token
2. Calculate cosine similarity matrix
3. Find optimal token alignment
4. Aggregate scores

**Scores Reported**:
- Precision: Generated semantics accuracy
- Recall: Reference semantics coverage
- F1: Overall semantic similarity

**Advantages Over ROUGE**:
- Captures semantic meaning
- Handles paraphrasing
- Language-aware

**Interpretation**:
- **< 0.7**: Poor semantic match
- **0.7-0.8**: Acceptable match
- **0.8-0.9**: Good match
- **> 0.9**: Excellent match

**Usage**:
```python
from src.evaluation import BERTScoreMetric

bertscore = BERTScoreMetric(model="vinai/phobert-base")
scores = bertscore.compute(
    predictions=["Response 1", "Response 2"],
    references=["Reference 1", "Reference 2"]
)
```

### 3. RAGAS Metrics

**Purpose**: RAG-specific quality assessment

**Metrics**:

#### Faithfulness
- Measures: Response consistency with retrieved context
- Score: [0, 1]
- Formula: (Supported statements) / (Total statements)

**Example**:
- Context: "Lãi suất 6.5%"
- Response: "Lãi suất là 6.5%" → High faithfulness
- Response: "Lãi suất là 7%" → Low faithfulness

#### Answer Relevancy
- Measures: Response relevance to query
- Uses: Cosine similarity between query and response embeddings
- Score: [0, 1]

**Example**:
- Query: "Lãi suất tiết kiệm?"
- Response: "Lãi suất là 6.5%" → High relevancy
- Response: "Chúng tôi có nhiều sản phẩm" → Low relevancy

#### Context Precision
- Measures: Quality of retrieved documents
- Formula: Relevant docs in top-K / K
- Score: [0, 1]

**Example**:
- Retrieved 5 docs, 4 relevant → Precision = 0.8

#### Context Recall (requires ground truth)
- Measures: Coverage of relevant information
- Formula: Retrieved relevant docs / All relevant docs
- Score: [0, 1]

**Usage**:
```python
from src.evaluation import RAGASMetric

ragas = RAGASMetric()
scores = ragas.compute(
    query="Lãi suất tiết kiệm?",
    response="Lãi suất là 6.5%/năm",
    contexts=["Context 1", "Context 2"]
)
```

### 4. Simple Heuristic Metrics

**Response Length**:
- Count: Number of words
- Too short (< 10): Likely incomplete
- Too long (> 200): Possibly verbose

**Context Overlap**:
- Formula: (Response ∩ Context) / Response words
- High overlap (> 0.7): Good grounding
- Low overlap (< 0.3): May be hallucinating

**Query Presence**:
- Check: Query keywords in response
- Binary: True/False

## Evaluation Pipeline

### Auto-Evaluation Process

**Sampling**:
- Rate: 10% of conversations (configurable)
- Min samples: 10
- Max samples: 100
- Selection: Random sampling

**Frequency**:
- Schedule: Every 24 hours
- Trigger: Manual or automatic

**Workflow**:
```
1. Sample conversations from MongoDB
   ↓
2. Extract query, response, contexts
   ↓
3. Compute all metrics
   ↓
4. Aggregate results
   ↓
5. Identify low-quality responses
   ↓
6. Generate report
```

### Low Quality Identification

**Criteria**:
- Context overlap < 0.5
- Response length < 10 words
- Faithfulness < 0.6 (if RAGAS enabled)
- User feedback rating < 3

**Actions**:
- Flag for manual review
- Trigger retraining
- Update prompts
- Add to training data

## Running Evaluation

### Manual Evaluation

```python
from src.evaluation import create_auto_evaluator, create_logger

# Setup
conv_logger = create_logger(use_mongodb=True)
auto_eval = create_auto_evaluator(
    conv_logger,
    enable_rouge=True,
    enable_bertscore=False,  # Slow, optional
    enable_ragas=False,      # Requires OpenAI
    sample_rate=0.1          # 10%
)

# Run evaluation
report = auto_eval.run_evaluation(days=7)

# View results
print(f"Status: {report['status']}")
print(f"Total Evaluated: {report['total_evaluated']}")
print(f"Low Quality Rate: {report['low_quality_rate']:.2%}")
print(f"\nAggregated Metrics:")
for metric, value in report['aggregated_metrics'].items():
    print(f"  {metric}: {value:.4f}")
```

### Scheduled Evaluation

```python
from src.evaluation import create_auto_evaluator, EvaluationScheduler

# Setup
auto_eval = create_auto_evaluator(conv_logger)
scheduler = EvaluationScheduler(
    auto_evaluator=auto_eval,
    interval_hours=24
)

# Run periodically
while True:
    report = scheduler.run()
    if report:
        print(f"Evaluation completed: {report['evaluated_at']}")
    time.sleep(3600)  # Check every hour
```

## Interpreting Results

### Metric Targets

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| ROUGE-1 F1 | < 0.3 | 0.3-0.5 | 0.5-0.7 | > 0.7 |
| ROUGE-2 F1 | < 0.2 | 0.2-0.3 | 0.3-0.5 | > 0.5 |
| BERTScore F1 | < 0.7 | 0.7-0.8 | 0.8-0.9 | > 0.9 |
| Faithfulness | < 0.6 | 0.6-0.7 | 0.7-0.85 | > 0.85 |
| Relevancy | < 0.6 | 0.6-0.75 | 0.75-0.9 | > 0.9 |
| Context Overlap | < 0.3 | 0.3-0.5 | 0.5-0.7 | > 0.7 |

### Report Structure

```json
{
  "status": "success",
  "evaluated_at": "2025-12-11T...",
  "period_days": 7,
  "total_sampled": 50,
  "total_evaluated": 48,
  "aggregated_metrics": {
    "rouge1_f_mean": 0.65,
    "rouge1_f_std": 0.12,
    "context_overlap_mean": 0.58,
    "response_length_mean": 45.3
  },
  "low_quality_count": 5,
  "low_quality_rate": 0.104,
  "low_quality_samples": [...]
}
```

## Quality Improvement Workflow

### 1. Identify Issues
```python
# Run evaluation
report = auto_eval.run_evaluation()

# Check low quality samples
for sample in report['low_quality_samples']:
    print(f"Query: {sample.get('query')}")
    print(f"Response: {sample.get('response')}")
    print(f"Context Overlap: {sample.get('context_overlap')}")
```

### 2. Analyze Root Causes
- **Low context overlap**: Poor retrieval, need more documents
- **Short responses**: LLM not generating enough, adjust prompts
- **Low faithfulness**: Hallucination, improve context quality

### 3. Apply Fixes
- Add more training documents
- Adjust retrieval threshold
- Update prompt templates
- Fine-tune LLM parameters

### 4. Re-evaluate
- Run evaluation again
- Compare before/after metrics
- Iterate until satisfactory

## Best Practices

1. **Regular Monitoring**: Run evaluation weekly
2. **Diverse Sampling**: Ensure representation across query types
3. **Multiple Metrics**: Use combination of ROUGE, BERTScore, heuristics
4. **Human Review**: Validate low-quality samples manually
5. **Continuous Improvement**: Track metrics over time
6. **User Feedback**: Correlate with actual user ratings

## Limitations

**ROUGE**:
- Only measures lexical overlap
- Doesn't understand semantics
- Biased towards reference phrasing

**BERTScore**:
- Slower than ROUGE
- Requires Vietnamese-specific model
- May not capture domain nuances

**RAGAS**:
- Requires OpenAI API (cost)
- Depends on LLM quality
- No Vietnamese-specific implementation

**Heuristics**:
- Oversimplified
- May miss nuanced quality issues
- Need domain-specific tuning

## Future Enhancements

- [ ] Vietnamese-specific RAGAS implementation
- [ ] User feedback integration
- [ ] A/B testing framework
- [ ] Real-time quality monitoring
- [ ] Automated retraining triggers
- [ ] Domain-specific metrics (banking terminology accuracy)
