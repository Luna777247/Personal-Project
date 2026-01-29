"""
Automatic evaluation pipeline for RAG responses
"""
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import random

from .logger import ConversationLogger
from .metrics import ResponseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoEvaluator:
    """Automatic evaluation pipeline"""
    
    def __init__(
        self,
        conversation_logger: ConversationLogger,
        evaluator: ResponseEvaluator,
        sample_rate: float = 0.1,
        min_samples: int = 10,
        max_samples: int = 100
    ):
        """
        Initialize auto evaluator
        
        Args:
            conversation_logger: Conversation logger instance
            evaluator: Response evaluator instance
            sample_rate: Sampling rate (default 10%)
            min_samples: Minimum samples to evaluate
            max_samples: Maximum samples to evaluate
        """
        self.conv_logger = conversation_logger
        self.evaluator = evaluator
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    def sample_conversations(
        self,
        days: int = 7,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Sample conversations for evaluation
        
        Args:
            days: Number of days to look back
            session_id: Optional session filter
            user_id: Optional user filter
            
        Returns:
            Sampled conversations
        """
        # Get all conversations
        conversations = self.conv_logger.get_conversations(
            session_id=session_id,
            user_id=user_id,
            limit=10000  # Large limit to get all
        )
        
        if not conversations:
            logger.warning("No conversations found for sampling")
            return []
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_convs = [
            c for c in conversations
            if c.get('timestamp', datetime.min) >= cutoff_date
        ]
        
        logger.info(f"Found {len(recent_convs)} conversations in last {days} days")
        
        # Calculate sample size
        sample_size = max(
            self.min_samples,
            min(int(len(recent_convs) * self.sample_rate), self.max_samples)
        )
        
        # Random sampling
        if len(recent_convs) <= sample_size:
            sampled = recent_convs
        else:
            sampled = random.sample(recent_convs, sample_size)
        
        logger.info(f"Sampled {len(sampled)} conversations for evaluation")
        
        return sampled
    
    def evaluate_conversations(
        self,
        conversations: List[Dict],
        include_reference: bool = False
    ) -> List[Dict]:
        """
        Evaluate sampled conversations
        
        Args:
            conversations: List of conversations
            include_reference: Whether to include reference answers
            
        Returns:
            Evaluation results
        """
        results = []
        
        for conv in conversations:
            try:
                # Extract data
                query = conv.get('query', '')
                response = conv.get('response', '')
                retrieved_docs = conv.get('retrieved_docs', [])
                
                # Extract contexts
                contexts = [doc.get('content_preview', '') for doc in retrieved_docs]
                
                # Evaluate
                eval_result = self.evaluator.evaluate(
                    query=query,
                    response=response,
                    contexts=contexts,
                    reference=None  # No ground truth available
                )
                
                # Add metadata
                eval_result.update({
                    'session_id': conv.get('session_id'),
                    'user_id': conv.get('user_id'),
                    'timestamp': conv.get('timestamp'),
                    'provider': conv.get('provider'),
                    'model': conv.get('model'),
                    'evaluated_at': datetime.now()
                })
                
                results.append(eval_result)
            
            except Exception as e:
                logger.error(f"Evaluation failed for conversation: {e}")
                continue
        
        logger.info(f"Evaluated {len(results)} conversations")
        
        return results
    
    def identify_low_quality(
        self,
        results: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Identify low quality responses
        
        Args:
            results: Evaluation results
            threshold: Quality threshold
            
        Returns:
            Low quality responses
        """
        low_quality = []
        
        for result in results:
            # Check multiple quality indicators
            context_overlap = result.get('context_overlap', 0)
            response_length = result.get('response_length', 0)
            
            # Low quality criteria
            if (
                context_overlap < threshold or
                response_length < 10
            ):
                low_quality.append(result)
        
        logger.info(f"Found {len(low_quality)} low quality responses")
        
        return low_quality
    
    def run_evaluation(
        self,
        days: int = 7,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            days: Number of days to look back
            session_id: Optional session filter
            user_id: Optional user filter
            
        Returns:
            Evaluation report
        """
        logger.info("Starting auto evaluation pipeline")
        
        # Sample conversations
        sampled = self.sample_conversations(days, session_id, user_id)
        
        if not sampled:
            return {
                'status': 'no_data',
                'message': 'No conversations found for evaluation'
            }
        
        # Evaluate
        results = self.evaluate_conversations(sampled)
        
        if not results:
            return {
                'status': 'failed',
                'message': 'Evaluation failed'
            }
        
        # Aggregate results
        aggregated = self.evaluator.aggregate_results(results)
        
        # Identify low quality
        low_quality = self.identify_low_quality(results)
        
        # Generate report
        report = {
            'status': 'success',
            'evaluated_at': datetime.now(),
            'period_days': days,
            'total_sampled': len(sampled),
            'total_evaluated': len(results),
            'aggregated_metrics': aggregated,
            'low_quality_count': len(low_quality),
            'low_quality_rate': len(low_quality) / len(results) if results else 0,
            'low_quality_samples': low_quality[:5]  # First 5 samples
        }
        
        logger.info("Auto evaluation completed successfully")
        
        return report


class EvaluationScheduler:
    """Scheduler for periodic evaluation"""
    
    def __init__(
        self,
        auto_evaluator: AutoEvaluator,
        interval_hours: int = 24
    ):
        """
        Initialize scheduler
        
        Args:
            auto_evaluator: Auto evaluator instance
            interval_hours: Evaluation interval in hours
        """
        self.auto_evaluator = auto_evaluator
        self.interval_hours = interval_hours
        self.last_run = None
    
    def should_run(self) -> bool:
        """Check if evaluation should run"""
        if not self.last_run:
            return True
        
        elapsed = datetime.now() - self.last_run
        
        return elapsed.total_seconds() >= self.interval_hours * 3600
    
    def run(self) -> Optional[Dict]:
        """Run scheduled evaluation"""
        if not self.should_run():
            logger.info("Skipping evaluation - not time yet")
            return None
        
        logger.info("Running scheduled evaluation")
        
        report = self.auto_evaluator.run_evaluation()
        
        self.last_run = datetime.now()
        
        return report


def create_auto_evaluator(
    conversation_logger: ConversationLogger,
    enable_rouge: bool = True,
    enable_bertscore: bool = False,
    enable_ragas: bool = False,
    sample_rate: float = 0.1
) -> AutoEvaluator:
    """
    Factory function to create auto evaluator
    
    Args:
        conversation_logger: Conversation logger instance
        enable_rouge: Enable ROUGE metrics
        enable_bertscore: Enable BERTScore
        enable_ragas: Enable RAGAS metrics
        sample_rate: Sampling rate
        
    Returns:
        Auto evaluator instance
    """
    evaluator = ResponseEvaluator(
        enable_rouge=enable_rouge,
        enable_bertscore=enable_bertscore,
        enable_ragas=enable_ragas
    )
    
    return AutoEvaluator(
        conversation_logger=conversation_logger,
        evaluator=evaluator,
        sample_rate=sample_rate
    )


if __name__ == "__main__":
    from .logger import create_logger
    
    # Test auto evaluation
    conv_logger = create_logger(use_mongodb=True)
    auto_eval = create_auto_evaluator(conv_logger, sample_rate=1.0)
    
    # Run evaluation
    report = auto_eval.run_evaluation(days=7)
    
    print("\nEvaluation Report:")
    print(f"Status: {report['status']}")
    print(f"Total Evaluated: {report.get('total_evaluated', 0)}")
    print(f"Low Quality Rate: {report.get('low_quality_rate', 0):.2%}")
    
    if 'aggregated_metrics' in report:
        print("\nAggregated Metrics:")
        for metric, value in report['aggregated_metrics'].items():
            print(f"  {metric}: {value:.4f}")
