"""
Evaluation metrics for RAG responses
"""
from typing import List, Dict, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROUGEMetric:
    """ROUGE metrics for response evaluation"""
    
    def __init__(self):
        """Initialize ROUGE metric"""
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            logger.info("ROUGE scorer initialized")
        except ImportError:
            logger.warning("rouge-score not installed")
            self.scorer = None
    
    def compute(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            ROUGE scores
        """
        if not self.scorer:
            return {}
        
        try:
            scores = self.scorer.score(reference, prediction)
            
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall,
            }
        
        except Exception as e:
            logger.error(f"ROUGE computation failed: {e}")
            return {}


class BERTScoreMetric:
    """BERTScore for semantic similarity"""
    
    def __init__(self, model: str = "vinai/phobert-base"):
        """
        Initialize BERTScore
        
        Args:
            model: Model name for BERTScore
        """
        self.model = model
        
        try:
            import bert_score
            self.bert_score = bert_score
            logger.info(f"BERTScore initialized with model: {model}")
        except ImportError:
            logger.warning("bert-score not installed")
            self.bert_score = None
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore
        
        Args:
            predictions: List of predictions
            references: List of references
            
        Returns:
            BERTScore metrics
        """
        if not self.bert_score:
            return {}
        
        try:
            P, R, F1 = self.bert_score.score(
                predictions,
                references,
                model_type=self.model,
                verbose=False
            )
            
            return {
                'bertscore_precision': float(P.mean()),
                'bertscore_recall': float(R.mean()),
                'bertscore_f1': float(F1.mean())
            }
        
        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {}


class RAGASMetric:
    """RAGAS metrics for RAG evaluation"""
    
    def __init__(self):
        """Initialize RAGAS metrics"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            
            self.evaluate = evaluate
            self.metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
            
            logger.info("RAGAS metrics initialized")
        
        except ImportError:
            logger.warning("ragas not installed")
            self.evaluate = None
            self.metrics = {}
    
    def compute(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute RAGAS metrics
        
        Args:
            query: User query
            response: Generated response
            contexts: Retrieved contexts
            ground_truth: Ground truth answer (optional)
            
        Returns:
            RAGAS scores
        """
        if not self.evaluate:
            return {}
        
        try:
            from datasets import Dataset
            
            # Prepare data
            data = {
                'question': [query],
                'answer': [response],
                'contexts': [contexts],
            }
            
            if ground_truth:
                data['ground_truth'] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Evaluate
            result = self.evaluate(
                dataset,
                metrics=list(self.metrics.values())
            )
            
            return {
                'ragas_faithfulness': result.get('faithfulness', 0.0),
                'ragas_answer_relevancy': result.get('answer_relevancy', 0.0),
                'ragas_context_precision': result.get('context_precision', 0.0),
                'ragas_context_recall': result.get('context_recall', 0.0) if ground_truth else None
            }
        
        except Exception as e:
            logger.error(f"RAGAS computation failed: {e}")
            return {}


class SimpleMetrics:
    """Simple heuristic metrics"""
    
    @staticmethod
    def response_length(text: str) -> int:
        """Count response length in words"""
        return len(text.split())
    
    @staticmethod
    def contains_keywords(text: str, keywords: List[str]) -> bool:
        """Check if text contains keywords"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)
    
    @staticmethod
    def context_overlap(response: str, contexts: List[str]) -> float:
        """
        Calculate overlap between response and contexts
        
        Args:
            response: Generated response
            contexts: Retrieved contexts
            
        Returns:
            Overlap ratio
        """
        response_words = set(response.lower().split())
        
        if not response_words:
            return 0.0
        
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(response_words & context_words)
        
        return overlap / len(response_words)
    
    @staticmethod
    def relevance_score(
        response: str,
        query: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """
        Simple relevance scoring
        
        Args:
            response: Generated response
            query: User query
            contexts: Retrieved contexts
            
        Returns:
            Relevance scores
        """
        return {
            'response_length': SimpleMetrics.response_length(response),
            'context_overlap': SimpleMetrics.context_overlap(response, contexts),
            'query_in_response': query.lower() in response.lower()
        }


class ResponseEvaluator:
    """Comprehensive response evaluator"""
    
    def __init__(
        self,
        enable_rouge: bool = True,
        enable_bertscore: bool = True,
        enable_ragas: bool = False  # Disabled by default (requires OpenAI API)
    ):
        """
        Initialize evaluator
        
        Args:
            enable_rouge: Enable ROUGE metrics
            enable_bertscore: Enable BERTScore
            enable_ragas: Enable RAGAS metrics
        """
        self.rouge = ROUGEMetric() if enable_rouge else None
        self.bertscore = BERTScoreMetric() if enable_bertscore else None
        self.ragas = RAGASMetric() if enable_ragas else None
        self.simple = SimpleMetrics()
    
    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            query: User query
            response: Generated response
            contexts: Retrieved contexts
            reference: Reference answer (optional)
            
        Returns:
            All evaluation metrics
        """
        results = {}
        
        # Simple metrics
        results.update(self.simple.relevance_score(response, query, contexts))
        
        # ROUGE
        if self.rouge and reference:
            results.update(self.rouge.compute(response, reference))
        
        # BERTScore
        if self.bertscore and reference:
            bertscore_result = self.bertscore.compute([response], [reference])
            results.update(bertscore_result)
        
        # RAGAS
        if self.ragas:
            ragas_result = self.ragas.compute(query, response, contexts, reference)
            results.update(ragas_result)
        
        logger.info(f"Evaluation completed with {len(results)} metrics")
        
        return results
    
    def batch_evaluate(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Batch evaluation
        
        Args:
            queries: List of queries
            responses: List of responses
            contexts_list: List of context lists
            references: List of reference answers
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, (query, response, contexts) in enumerate(zip(queries, responses, contexts_list)):
            reference = references[i] if references else None
            
            eval_result = self.evaluate(query, response, contexts, reference)
            results.append(eval_result)
        
        logger.info(f"Batch evaluation completed for {len(results)} samples")
        
        return results
    
    def aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for result in results:
            metric_names.update(result.keys())
        
        # Calculate averages
        for metric in metric_names:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)
        
        return aggregated


if __name__ == "__main__":
    # Test evaluation
    evaluator = ResponseEvaluator(enable_rouge=True, enable_bertscore=False)
    
    query = "Lãi suất tiết kiệm MB Bank là bao nhiêu?"
    response = "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là 6.0%/năm, kỳ hạn 12 tháng là 6.5%/năm."
    contexts = [
        "Lãi suất tiết kiệm MB Bank: Kỳ hạn 6 tháng 6.0%/năm, kỳ hạn 12 tháng 6.5%/năm.",
        "MB Bank cung cấp nhiều sản phẩm tiết kiệm với lãi suất hấp dẫn."
    ]
    reference = "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là 6.0% và 12 tháng là 6.5%."
    
    results = evaluator.evaluate(query, response, contexts, reference)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
