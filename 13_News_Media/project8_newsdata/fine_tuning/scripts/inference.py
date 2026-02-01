#!/usr/bin/env python3
"""
Inference Script for Fine-tuned Disaster Information Extraction Models
Production-ready inference pipeline for disaster information extraction
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterInformationExtractor:
    """Production-ready disaster information extractor using fine-tuned models"""

    def __init__(self, models_dir: str):
        """
        Initialize the extractor with trained models

        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.tokenizers = {}
        self.label_mappings = {}

        # Load all available models
        self._load_models()

    def _load_models(self):
        """Load all trained models"""
        model_types = ["ner", "event_extraction", "relation_extraction"]

        for model_type in model_types:
            model_path = self.models_dir / f"{model_type}_model"
            if model_path.exists():
                try:
                    logger.info(f"Loading {model_type} model from {model_path}")

                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    self.tokenizers[model_type] = tokenizer

                    # Load model
                    if model_type == "ner":
                        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

                    self.models[model_type] = model

                    # Load label mappings
                    if model_type == "ner":
                        with open(model_path / "label2id.json", 'r') as f:
                            self.label_mappings[model_type] = json.load(f)
                    else:
                        with open(model_path / "id2label.json", 'r') as f:
                            self.label_mappings[model_type] = json.load(f)

                    # Move to device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)
                    model.eval()

                    logger.info(f"Successfully loaded {model_type} model")

                except Exception as e:
                    logger.warning(f"Failed to load {model_type} model: {e}")
            else:
                logger.warning(f"Model directory not found: {model_path}")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract disaster-related entities from text using NER model

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        if "ner" not in self.models:
            logger.warning("NER model not available")
            return []

        tokenizer = self.tokenizers["ner"]
        model = self.models["ner"]
        id2label = {int(k): v for k, v in self.label_mappings["ner"].items()}

        device = next(model.parameters()).device

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

        # Extract entities
        entities = []
        current_entity = None

        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]  # Remove B-
                current_entity = {
                    "type": entity_type,
                    "text": token.replace("##", ""),
                    "start": encoding.token_to_chars(i)[0] if encoding.token_to_chars(i) else 0,
                    "end": encoding.token_to_chars(i)[1] if encoding.token_to_chars(i) else len(token),
                    "confidence": 1.0  # Placeholder
                }

            elif label.startswith("I-") and current_entity:
                # Continuation of current entity
                current_entity["text"] += token.replace("##", "")
                current_entity["end"] = encoding.token_to_chars(i)[1] if encoding.token_to_chars(i) else current_entity["end"] + len(token)

            elif current_entity:
                # End of current entity
                entities.append(current_entity)
                current_entity = None

        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    def classify_event(self, text: str) -> Dict[str, Any]:
        """
        Classify disaster event type from text

        Args:
            text: Input text

        Returns:
            Event classification result
        """
        if "event_extraction" not in self.models:
            logger.warning("Event extraction model not available")
            return {"event_type": "UNKNOWN", "confidence": 0.0}

        tokenizer = self.tokenizers["event_extraction"]
        model = self.models["event_extraction"]
        id2label = {int(k): v for k, v in self.label_mappings["event_extraction"].items()}

        device = next(model.parameters()).device

        # Tokenize
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(predictions, dim=1)

        event_type = id2label[pred_class.item()]
        confidence = predictions[0][pred_class.item()].item()

        return {
            "event_type": event_type,
            "confidence": confidence
        }

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations between entities

        Args:
            text: Input text
            entities: List of entities extracted from text

        Returns:
            List of relations between entities
        """
        if "relation_extraction" not in self.models:
            logger.warning("Relation extraction model not available")
            return []

        tokenizer = self.tokenizers["relation_extraction"]
        model = self.models["relation_extraction"]
        id2label = {int(k): v for k, v in self.label_mappings["relation_extraction"].items()}

        device = next(model.parameters()).device

        relations = []

        # Generate all entity pairs
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue

                # Create marked text
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

                # Tokenize
                encoding = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=512)
                encoding = {k: v.to(device) for k, v in encoding.items()}

                # Predict
                with torch.no_grad():
                    outputs = model(**encoding)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    pred_class = torch.argmax(predictions, dim=1)

                relation_type = id2label[pred_class.item()]
                confidence = predictions[0][pred_class.item()].item()

                if relation_type != "NO_RELATION":
                    relations.append({
                        "head": head,
                        "tail": tail,
                        "relation_type": relation_type,
                        "confidence": confidence
                    })

        return relations

    def extract_disaster_info(self, text: str) -> Dict[str, Any]:
        """
        Complete disaster information extraction pipeline

        Args:
            text: Input disaster news text

        Returns:
            Complete disaster information structure
        """
        logger.info("Starting disaster information extraction...")

        # Step 1: Extract entities
        entities = self.extract_entities(text)
        logger.info(f"Extracted {len(entities)} entities")

        # Step 2: Classify event type
        event_info = self.classify_event(text)
        logger.info(f"Classified event type: {event_info['event_type']} (confidence: {event_info['confidence']:.3f})")

        # Step 3: Extract relations
        relations = self.extract_relations(text, entities)
        logger.info(f"Extracted {len(relations)} relations")

        # Step 4: Structure the information
        disaster_info = {
            "event_type": event_info["event_type"],
            "event_confidence": event_info["confidence"],
            "entities": entities,
            "relations": relations,
            "text": text
        }

        # Add structured information based on entities
        disaster_info.update(self._structure_entities(entities))

        return disaster_info

    def _structure_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structure entities into disaster information categories"""
        structured = {
            "disaster_type": None,
            "location": None,
            "time": None,
            "damage": None,
            "response": None,
            "impact": None,
            "forecast": None
        }

        for entity in entities:
            entity_type = entity["type"].lower()
            entity_text = entity["text"]

            if entity_type == "disaster_type":
                structured["disaster_type"] = entity_text
            elif entity_type == "location":
                if structured["location"] is None:
                    structured["location"] = []
                structured["location"].append(entity_text)
            elif entity_type == "time":
                structured["time"] = entity_text
            elif entity_type == "damage":
                if structured["damage"] is None:
                    structured["damage"] = []
                structured["damage"].append(entity_text)
            elif entity_type == "response":
                if structured["response"] is None:
                    structured["response"] = []
                structured["response"].append(entity_text)
            elif entity_type == "impact":
                if structured["impact"] is None:
                    structured["impact"] = []
                structured["impact"].append(entity_text)
            elif entity_type == "forecast":
                structured["forecast"] = entity_text

        return structured

    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract disaster information from multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of disaster information structures
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            try:
                result = self.extract_disaster_info(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i+1}: {e}")
                results.append({
                    "error": str(e),
                    "text": text
                })

        return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Disaster Information Extraction Inference")
    parser.add_argument("--models-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--input", required=True, help="Input text or file path")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--batch", action="store_true", help="Process multiple texts from file")

    args = parser.parse_args()

    # Initialize extractor
    extractor = DisasterInformationExtractor(args.models_dir)

    if args.batch:
        # Process multiple texts from file
        with open(args.input, 'r', encoding='utf-8') as f:
            if args.input.endswith('.json'):
                texts = json.load(f)
                if isinstance(texts, list):
                    input_texts = texts
                else:
                    input_texts = [texts.get("text", "")]
            else:
                input_texts = [line.strip() for line in f if line.strip()]

        results = extractor.batch_extract(input_texts)

    else:
        # Process single text
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.input

        results = extractor.extract_disaster_info(text)

    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()