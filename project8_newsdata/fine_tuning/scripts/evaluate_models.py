#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Disaster Information Extraction Models
Comprehensive evaluation with multiple metrics and visualizations
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(data_path: str, task: str) -> List[Dict[str, Any]]:
    """Load test data based on task type"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if task == "ner":
        return data  # Already in correct format
    elif task == "event_extraction":
        return data  # Already in correct format
    elif task == "relation_extraction":
        return load_relation_data(data_path)  # Convert to relation instances
    else:
        raise ValueError(f"Unknown task: {task}")


def load_relation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load relation extraction test data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to relation instances (same as in training)
    relation_instances = []

    for doc in data:
        text = doc["text"]
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])

        # Create entity lookup
        entity_lookup = {f"{e['start']}-{e['end']}": e for e in entities}

        # Process existing relations
        for rel in relations:
            head_key = f"{rel['head']['start']}-{rel['head']['end']}"
            tail_key = f"{rel['tail']['start']}-{rel['tail']['end']}"

            if head_key in entity_lookup and tail_key in entity_lookup:
                relation_instances.append({
                    "text": text,
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "relation_type": rel["relation_type"]
                })

        # Generate negative examples (no relation)
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i != j:
                    # Check if this pair already has a relation
                    has_relation = any(
                        (r["head"]["start"] == head["start"] and r["head"]["end"] == head["end"] and
                         r["tail"]["start"] == tail["start"] and r["tail"]["end"] == tail["end"])
                        for r in relations
                    )

                    if not has_relation:
                        relation_instances.append({
                            "text": text,
                            "head": head,
                            "tail": tail,
                            "relation_type": "NO_RELATION"
                        })

    return relation_instances


def evaluate_ner_model(model_path: str, test_data: List[Dict[str, Any]], output_dir: str):
    """Evaluate NER model"""
    logger.info("Evaluating NER model...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Load label mappings
    with open(os.path.join(model_path, "label2id.json"), 'r') as f:
        label2id = json.load(f)

    id2label = {int(k): v for k, v in model["id2label"].items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    for item in test_data:
        text = item["text"]
        true_entities = item["entities"]

        # Tokenize
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Convert predictions to entities
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        pred_labels = [id2label[pred.item()] for pred in predictions[0]]

        # Convert true entities to label sequence
        true_labels = ["O"] * len(tokens)
        for entity in true_entities:
            start_char = entity["start"]
            end_char = entity["end"]
            label = entity["label"]

            # Find tokens that correspond to the entity
            token_start = encoding.char_to_token(start_char)
            token_end = encoding.char_to_token(end_char - 1)

            if token_start is None or token_end is None:
                continue

            # Set B- label for first token
            if token_start < len(true_labels):
                true_labels[token_start] = f"B-{label}"

            # Set I- label for subsequent tokens
            for i in range(token_start + 1, min(token_end + 1, len(true_labels))):
                true_labels[i] = f"I-{label}"

        # Align predictions and labels (handle subword tokens)
        aligned_predictions = []
        aligned_labels = []

        for pred, true in zip(pred_labels, true_labels):
            if not tokens[len(aligned_predictions)].startswith("##"):
                aligned_predictions.append(pred)
                aligned_labels.append(true)

        all_predictions.extend(aligned_predictions)
        all_labels.extend(aligned_labels)

    # Calculate metrics
    report = classification_report(all_labels, all_predictions, output_dict=True)
    cm = confusion_matrix(all_labels, all_predictions)

    # Save results
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": accuracy_score(all_labels, all_predictions)
    }

    with open(os.path.join(output_dir, "ner_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    create_confusion_matrix_plot(cm, list(id2label.values()), os.path.join(output_dir, "ner_confusion_matrix.png"))
    create_classification_report_plot(report, os.path.join(output_dir, "ner_classification_report.png"))

    logger.info(f"NER Evaluation completed. F1: {results['weighted_f1']:.3f}")

    return results


def evaluate_event_extraction_model(model_path: str, test_data: List[Dict[str, Any]], output_dir: str):
    """Evaluate Event Extraction model"""
    logger.info("Evaluating Event Extraction model...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load label mappings
    with open(os.path.join(model_path, "id2label.json"), 'r') as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions = []
    labels = []

    for item in test_data:
        text = item["text"]
        true_label = item.get("event_type", "O")

        # Tokenize
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            pred = torch.argmax(outputs.logits, dim=1)

        pred_label = id2label[pred.item()]
        predictions.append(pred_label)
        labels.append(true_label)

    # Calculate metrics
    report = classification_report(labels, predictions, output_dict=True)
    cm = confusion_matrix(labels, predictions)

    # Save results
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": accuracy_score(labels, predictions)
    }

    with open(os.path.join(output_dir, "event_extraction_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    create_confusion_matrix_plot(cm, list(id2label.values()), os.path.join(output_dir, "event_confusion_matrix.png"))
    create_classification_report_plot(report, os.path.join(output_dir, "event_classification_report.png"))

    logger.info(f"Event Extraction Evaluation completed. F1: {results['weighted_f1']:.3f}")

    return results


def evaluate_relation_extraction_model(model_path: str, test_data: List[Dict[str, Any]], output_dir: str):
    """Evaluate Relation Extraction model"""
    logger.info("Evaluating Relation Extraction model...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load label mappings
    with open(os.path.join(model_path, "id2label.json"), 'r') as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions = []
    labels = []

    for item in test_data:
        # Create input text with entity markers
        text = item["text"]
        head = item["head"]
        tail = item["tail"]

        # Mark entities in text
        marked_text = text
        offset = 0

        # Insert tail entity marker first (higher position)
        tail_start = tail["start"] + offset
        tail_end = tail["end"] + offset
        marked_text = marked_text[:tail_start] + "[TAIL] " + marked_text[tail_start:tail_end] + " [/TAIL]" + marked_text[tail_end:]
        offset += len("[TAIL]  [/TAIL]")

        # Insert head entity marker
        head_start = head["start"] + offset
        head_end = head["end"] + offset
        marked_text = marked_text[:head_start] + "[HEAD] " + marked_text[head_start:head_end] + " [/HEAD]" + marked_text[head_end:]

        true_label = item.get("relation_type", "NO_RELATION")

        # Tokenize
        encoding = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            pred = torch.argmax(outputs.logits, dim=1)

        pred_label = id2label[pred.item()]
        predictions.append(pred_label)
        labels.append(true_label)

    # Calculate metrics
    report = classification_report(labels, predictions, output_dict=True)
    cm = confusion_matrix(labels, predictions)

    # Save results
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "accuracy": accuracy_score(labels, predictions)
    }

    with open(os.path.join(output_dir, "relation_extraction_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    create_confusion_matrix_plot(cm, list(id2label.values()), os.path.join(output_dir, "relation_confusion_matrix.png"))
    create_classification_report_plot(report, os.path.join(output_dir, "relation_classification_report.png"))

    logger.info(f"Relation Extraction Evaluation completed. F1: {results['weighted_f1']:.3f}")

    return results


def create_confusion_matrix_plot(cm: np.ndarray, labels: List[str], save_path: str):
    """Create confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_classification_report_plot(report: Dict[str, Any], save_path: str):
    """Create classification report visualization"""
    # Extract metrics for each class
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1 = [report[cls]['f1-score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Report by Class')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_report(results: Dict[str, Dict[str, Any]], output_dir: str):
    """Create comprehensive evaluation report"""
    report = {
        "evaluation_summary": {},
        "model_comparison": {},
        "recommendations": []
    }

    # Extract key metrics
    for model_type, metrics in results.items():
        report["evaluation_summary"][model_type] = {
            "accuracy": metrics.get("accuracy", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "weighted_f1": metrics.get("weighted_f1", 0)
        }

    # Model comparison
    report["model_comparison"] = {
        "best_performing": max(results.keys(), key=lambda x: results[x].get("weighted_f1", 0)),
        "worst_performing": min(results.keys(), key=lambda x: results[x].get("weighted_f1", 0)),
        "average_f1": np.mean([m.get("weighted_f1", 0) for m in results.values()])
    }

    # Generate recommendations
    for model_type, metrics in results.items():
        f1_score = metrics.get("weighted_f1", 0)
        if f1_score < 0.7:
            report["recommendations"].append(f"Improve {model_type} model (F1: {f1_score:.3f})")
        elif f1_score > 0.9:
            report["recommendations"].append(f"{model_type} model performing excellently (F1: {f1_score:.3f})")

    # Save comprehensive report
    with open(os.path.join(output_dir, "comprehensive_evaluation_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Create summary text
    summary_text = f"""
# Comprehensive Evaluation Report

## Model Performance Summary

"""
    for model_type, metrics in results.items():
        summary_text += f"### {model_type.upper()}\n"
        summary_text += f"- Accuracy: {metrics.get('accuracy', 0):.3f}\n"
        summary_text += f"- Macro F1: {metrics.get('macro_f1', 0):.3f}\n"
        summary_text += f"- Weighted F1: {metrics.get('weighted_f1', 0):.3f}\n\n"

    summary_text += "## Recommendations\n"
    for rec in report["recommendations"]:
        summary_text += f"- {rec}\n"

    with open(os.path.join(output_dir, "evaluation_summary.md"), 'w') as f:
        f.write(summary_text)

    logger.info("Comprehensive evaluation report created")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Disaster Extraction Models")
    parser.add_argument("--task", required=True, choices=["ner", "event_extraction", "relation_extraction", "all"],
                       help="Task to evaluate")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--output-dir", default="evaluation/", help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_data = load_test_data(args.test_data, args.task)

    results = {}

    if args.task == "ner":
        results["ner"] = evaluate_ner_model(args.model_path, test_data, args.output_dir)

    elif args.task == "event_extraction":
        results["event_extraction"] = evaluate_event_extraction_model(args.model_path, test_data, args.output_dir)

    elif args.task == "relation_extraction":
        results["relation_extraction"] = evaluate_relation_extraction_model(args.model_path, test_data, args.output_dir)

    elif args.task == "all":
        # Evaluate all models (assuming they are in subdirectories)
        model_base_path = args.model_path

        for task in ["ner", "event_extraction", "relation_extraction"]:
            model_path = os.path.join(model_base_path, f"{task}_model")
            if os.path.exists(model_path):
                if task == "ner":
                    results[task] = evaluate_ner_model(model_path, test_data, args.output_dir)
                elif task == "event_extraction":
                    results[task] = evaluate_event_extraction_model(model_path, test_data, args.output_dir)
                elif task == "relation_extraction":
                    results[task] = evaluate_relation_extraction_model(model_path, test_data, args.output_dir)

        # Create comprehensive report
        create_comprehensive_report(results, args.output_dir)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()