#!/usr/bin/env python3
"""
Complete Fine-tuning Pipeline Runner
Automated pipeline for training, evaluation, and deployment
"""

import os
import json
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FineTuningPipeline:
    """Complete fine-tuning pipeline manager"""

    def __init__(self, config_path: str, base_dir: str = "fine_tuning"):
        """
        Initialize pipeline

        Args:
            config_path: Path to configuration file
            base_dir: Base directory for fine-tuning
        """
        self.config_path = config_path
        self.base_dir = Path(base_dir)
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create run directory
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline with run directory: {self.run_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            import yaml
            return yaml.safe_load(f)

    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> bool:
        """Run a command and return success status"""
        try:
            logger.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd or str(self.base_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Command completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False

    def prepare_data(self) -> bool:
        """Prepare training data"""
        logger.info("Step 1: Preparing training data...")

        # Run annotation script
        annotate_cmd = [
            sys.executable, "scripts/annotate_data.py",
            "--input", str(self.base_dir / "data" / "raw"),
            "--output", str(self.run_dir / "data"),
            "--config", self.config_path
        ]

        if not self._run_command(annotate_cmd):
            logger.error("Data preparation failed")
            return False

        logger.info("Data preparation completed")
        return True

    def train_ner_model(self) -> bool:
        """Train NER model"""
        logger.info("Step 2: Training NER model...")

        train_cmd = [
            sys.executable, "scripts/train_ner.py",
            "--config", self.config_path,
            "--data", str(self.run_dir / "data" / "ner_train.json"),
            "--val-data", str(self.run_dir / "data" / "ner_val.json"),
            "--output", str(self.run_dir / "models" / "ner_model")
        ]

        if not self._run_command(train_cmd):
            logger.error("NER training failed")
            return False

        logger.info("NER training completed")
        return True

    def train_event_extraction_model(self) -> bool:
        """Train Event Extraction model"""
        logger.info("Step 3: Training Event Extraction model...")

        train_cmd = [
            sys.executable, "scripts/train_event_extraction.py",
            "--config", self.config_path,
            "--data", str(self.run_dir / "data" / "event_train.json"),
            "--val-data", str(self.run_dir / "data" / "event_val.json"),
            "--output", str(self.run_dir / "models" / "event_extraction_model")
        ]

        if not self._run_command(train_cmd):
            logger.error("Event Extraction training failed")
            return False

        logger.info("Event Extraction training completed")
        return True

    def train_relation_extraction_model(self) -> bool:
        """Train Relation Extraction model"""
        logger.info("Step 4: Training Relation Extraction model...")

        train_cmd = [
            sys.executable, "scripts/train_relation_extraction.py",
            "--config", self.config_path,
            "--data", str(self.run_dir / "data" / "relation_train.json"),
            "--val-data", str(self.run_dir / "data" / "relation_val.json"),
            "--output", str(self.run_dir / "models" / "relation_extraction_model")
        ]

        if not self._run_command(train_cmd):
            logger.error("Relation Extraction training failed")
            return False

        logger.info("Relation Extraction training completed")
        return True

    def evaluate_models(self) -> bool:
        """Evaluate all trained models"""
        logger.info("Step 5: Evaluating models...")

        eval_cmd = [
            sys.executable, "scripts/evaluate_models.py",
            "--task", "all",
            "--model-path", str(self.run_dir / "models"),
            "--test-data", str(self.run_dir / "data"),
            "--output-dir", str(self.run_dir / "evaluation")
        ]

        if not self._run_command(eval_cmd):
            logger.error("Model evaluation failed")
            return False

        logger.info("Model evaluation completed")
        return True

    def visualize_results(self) -> bool:
        """Create visualizations"""
        logger.info("Step 6: Creating visualizations...")

        viz_cmd = [
            sys.executable, "scripts/visualize_results.py",
            "--logs-dir", str(self.run_dir / "models"),
            "--evaluation-dir", str(self.run_dir / "evaluation"),
            "--output-dir", str(self.run_dir / "visualizations")
        ]

        if not self._run_command(viz_cmd):
            logger.error("Visualization creation failed")
            return False

        logger.info("Visualization creation completed")
        return True

    def create_deployment_package(self) -> bool:
        """Create deployment package"""
        logger.info("Step 7: Creating deployment package...")

        # Copy models to deployment directory
        deploy_dir = self.run_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)

        # Copy inference script
        import shutil
        shutil.copy(self.base_dir / "scripts" / "inference.py", deploy_dir)

        # Copy config
        shutil.copy(self.config_path, deploy_dir)

        # Create requirements for deployment
        deploy_requirements = [
            "torch>=1.9.0",
            "transformers>=4.21.0",
            "pyyaml>=6.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0"
        ]

        with open(deploy_dir / "requirements.txt", 'w') as f:
            f.write("\n".join(deploy_requirements))

        # Create deployment README
        deployment_readme = f"""
# Disaster Information Extraction - Deployment Package

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

1. Install dependencies:
   pip install -r requirements.txt

2. Run inference:
   python inference.py --models-dir models/ --input "your_disaster_news_text"

## Models Included

- NER Model: Named Entity Recognition for disaster entities
- Event Extraction Model: Disaster event type classification
- Relation Extraction Model: Entity relation extraction

## Configuration

See config.yaml for model parameters and settings.
"""

        with open(deploy_dir / "README.md", 'w') as f:
            f.write(deployment_readme)

        logger.info("Deployment package created")
        return True

    def run_full_pipeline(self, skip_steps: List[str] = None) -> bool:
        """
        Run the complete fine-tuning pipeline

        Args:
            skip_steps: List of steps to skip

        Returns:
            Success status
        """
        skip_steps = skip_steps or []

        steps = [
            ("prepare_data", "Data Preparation"),
            ("train_ner_model", "NER Training"),
            ("train_event_extraction_model", "Event Extraction Training"),
            ("train_relation_extraction_model", "Relation Extraction Training"),
            ("evaluate_models", "Model Evaluation"),
            ("visualize_results", "Results Visualization"),
            ("create_deployment_package", "Deployment Package Creation")
        ]

        success_count = 0
        total_steps = len(steps)

        for step_func, step_name in steps:
            if step_func in skip_steps:
                logger.info(f"Skipping {step_name}")
                continue

            logger.info(f"Starting {step_name} ({success_count + 1}/{total_steps})")

            if getattr(self, step_func)():
                success_count += 1
                logger.info(f"✓ {step_name} completed successfully")
            else:
                logger.error(f"✗ {step_name} failed")
                return False

        logger.info(f"Pipeline completed: {success_count}/{total_steps} steps successful")

        # Create summary report
        self._create_summary_report(success_count, total_steps)

        return success_count == total_steps

    def _create_summary_report(self, success_count: int, total_steps: int):
        """Create pipeline execution summary"""
        summary = {
            "timestamp": self.timestamp,
            "run_directory": str(self.run_dir),
            "steps_completed": success_count,
            "total_steps": total_steps,
            "success_rate": success_count / total_steps,
            "config_used": self.config_path,
            "models_trained": [
                "ner_model",
                "event_extraction_model",
                "relation_extraction_model"
            ] if success_count >= 4 else []
        }

        with open(self.run_dir / "pipeline_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable summary
        summary_text = f"""
# Fine-tuning Pipeline Summary

**Execution Time**: {self.timestamp}
**Run Directory**: {self.run_dir}
**Success Rate**: {success_count}/{total_steps} steps completed

## Completed Steps

"""

        steps_status = [
            ("Data Preparation", success_count >= 1),
            ("NER Training", success_count >= 2),
            ("Event Extraction Training", success_count >= 3),
            ("Relation Extraction Training", success_count >= 4),
            ("Model Evaluation", success_count >= 5),
            ("Results Visualization", success_count >= 6),
            ("Deployment Package Creation", success_count >= 7)
        ]

        for step, completed in steps_status:
            status = "✓" if completed else "✗"
            summary_text += f"- {status} {step}\n"

        summary_text += f"""

## Next Steps

1. Review evaluation results in `{self.run_dir}/evaluation/`
2. Check visualizations in `{self.run_dir}/visualizations/`
3. Use deployment package in `{self.run_dir}/deployment/` for production
4. Monitor model performance and retrain as needed

## Configuration Used

- Config file: {self.config_path}
- Models: {', '.join(summary['models_trained'])}
"""

        with open(self.run_dir / "pipeline_summary.md", 'w') as f:
            f.write(summary_text)

        logger.info(f"Summary report saved to {self.run_dir}/pipeline_summary.md")


def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description="Complete Fine-tuning Pipeline Runner")
    parser.add_argument("--config", default="fine_tuning/config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--base-dir", default="fine_tuning",
                       help="Base directory for fine-tuning")
    parser.add_argument("--skip-steps", nargs="+",
                       choices=["prepare_data", "train_ner_model", "train_event_extraction_model",
                               "train_relation_extraction_model", "evaluate_models",
                               "visualize_results", "create_deployment_package"],
                       help="Steps to skip in the pipeline")
    parser.add_argument("--step", choices=["prepare_data", "train_ner", "train_event",
                                         "train_relation", "evaluate", "visualize", "deploy"],
                       help="Run only a specific step")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = FineTuningPipeline(args.config, args.base_dir)

    if args.step:
        # Run single step
        step_mapping = {
            "prepare_data": pipeline.prepare_data,
            "train_ner": pipeline.train_ner_model,
            "train_event": pipeline.train_event_extraction_model,
            "train_relation": pipeline.train_relation_extraction_model,
            "evaluate": pipeline.evaluate_models,
            "visualize": pipeline.visualize_results,
            "deploy": pipeline.create_deployment_package
        }

        success = step_mapping[args.step]()
        if success:
            logger.info(f"Step '{args.step}' completed successfully")
        else:
            logger.error(f"Step '{args.step}' failed")
            sys.exit(1)

    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline(args.skip_steps)
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()