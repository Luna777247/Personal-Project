#!/usr/bin/env python3
"""
Data Annotation Tool for Disaster Information Extraction

Supports annotation for:
- Named Entity Recognition (NER)
- Event Extraction
- Relation Extraction

Entities: DISASTER, LOCATION, TIME, DAMAGE, RESPONSE, IMPACT, FORECAST
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    DISASTER = "DISASTER"
    LOCATION = "LOCATION"
    TIME = "TIME"
    DAMAGE = "DAMAGE"
    RESPONSE = "RESPONSE"
    IMPACT = "IMPACT"
    FORECAST = "FORECAST"


class RelationType(Enum):
    LOCATION_OF = "LOCATION_OF"
    TIME_OF = "TIME_OF"
    CAUSE_OF = "CAUSE_OF"
    IMPACT_OF = "IMPACT_OF"
    NO_RELATION = "NO_RELATION"


@dataclass
class Entity:
    """Named Entity"""
    start: int
    end: int
    text: str
    label: str
    confidence: float = 1.0


@dataclass
class Relation:
    """Relation between entities"""
    head: Entity
    tail: Entity
    relation_type: str
    confidence: float = 1.0


@dataclass
class AnnotatedDocument:
    """Annotated document with entities and relations"""
    id: str
    text: str
    entities: List[Entity]
    relations: List[Relation] = None
    event_type: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.relations is None:
            self.relations = []
        if self.metadata is None:
            self.metadata = {}


class DisasterAnnotator:
    """Automatic annotation tool for disaster information"""

    def __init__(self):
        # Disaster keywords
        self.disaster_keywords = {
            'bão': 'BAO',
            'storm': 'BAO',
            'hurricane': 'BAO',
            'typhoon': 'BAO',
            'lũ': 'LUU_LUT',
            'lụt': 'LUU_LUT',
            'flood': 'LUU_LUT',
            'ngập': 'LUU_LUT',
            'hạn hán': 'HAN_HAN',
            'drought': 'HAN_HAN',
            'khô hạn': 'HAN_HAN',
            'cháy rừng': 'CHAY_RUNG',
            'forest fire': 'CHAY_RUNG',
            'fire': 'CHAY_RUNG',
            'động đất': 'DONG_DAT',
            'earthquake': 'DONG_DAT',
            'sạt lở': 'SAT_LO',
            'landslide': 'SAT_LO',
            'luận': 'SAT_LO'
        }

        # Location patterns (Vietnamese provinces/cities)
        self.location_patterns = [
            r'\b(?:tỉnh|thành phố|tp\.?)\s+[A-ZĐ][a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+\b',
            r'\b[A-ZĐ][a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+(?:\s+[A-ZĐ][a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+)*\b'
        ]

        # Time patterns
        self.time_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY
            r'\b\d{1,2}\s+(?:tháng|giờ|ngày)\s+\d{1,2}\b',  # 15 tháng 11
            r'\b(?:hôm\s+qua|hôm\s+nay|ngày\s+mai)\b',  # yesterday, today, tomorrow
            r'\b\d{1,2}[:]\d{2}\b',  # HH:MM
            r'\b\d{1,2}\s+(?:giờ|sáng|chiều|tối)\b'  # 8 giờ sáng
        ]

        # Damage patterns
        self.damage_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:người|nghìn|triệu|tỷ)\s+(?:chết|thương|tổn thất|thiệt hại)\b',
            r'\b(?:hàng\s+)?(?:trăm|nghìn|triệu|tỷ)\s+(?:ngôi\s+nhà|căn\s+nhà|hecta?|mét|km)\b',
            r'\b\d+(?:\.\d+)?\s*(?:tỷ|triệu|nghìn)\s+(?:đồng|VNĐ)\b'
        ]

        # Response patterns
        self.response_patterns = [
            r'\b(?:UBND|cứu\s+hộ|cứu\s+trợ|chính\s+phủ|bộ\s+quốc\s+phòng)\b',
            r'\b(?:điều\s+động|phản\s+ứng|ứng\s+phó|giải\s+quyết)\b'
        ]

        # Impact patterns
        self.impact_patterns = [
            r'\b(?:ảnh\s+hưởng|bị\s+cắt|cách\s+ly|thiếu\s+thốn|khó\s+khăn)\b',
            r'\b\d+(?:\.\d+)?\s*(?:nghìn|triệu)\s+(?:hộ\s+dân|người\s+dân)\b'
        ]

        # Forecast patterns
        self.forecast_patterns = [
            r'\b(?:dự\s+báo|cảnh\s+báo|triển\s+vọng|có\s+khả\s+năng)\b',
            r'\b(?:sẽ|mạnh\s+hơn|tăng\s+cường|kéo\s+dài)\b'
        ]

    def annotate_text(self, text: str, doc_id: str = None) -> AnnotatedDocument:
        """Automatically annotate disaster-related text"""
        if doc_id is None:
            doc_id = f"doc_{hash(text) % 1000000}"

        entities = []
        entities.extend(self._extract_disaster_entities(text))
        entities.extend(self._extract_location_entities(text))
        entities.extend(self._extract_time_entities(text))
        entities.extend(self._extract_damage_entities(text))
        entities.extend(self._extract_response_entities(text))
        entities.extend(self._extract_impact_entities(text))
        entities.extend(self._extract_forecast_entities(text))

        # Remove overlapping entities (keep longer ones)
        entities = self._resolve_overlaps(entities)

        # Extract relations
        relations = self._extract_relations(entities)

        # Determine event type
        event_type = self._classify_event_type(text)

        return AnnotatedDocument(
            id=doc_id,
            text=text,
            entities=entities,
            relations=relations,
            event_type=event_type,
            metadata={"auto_annotated": True, "annotator_version": "1.0"}
        )

    def _extract_disaster_entities(self, text: str) -> List[Entity]:
        """Extract disaster entities"""
        entities = []
        text_lower = text.lower()

        for keyword, disaster_type in self.disaster_keywords.items():
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                start, end = match.span()
                entities.append(Entity(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=f"DISASTER-{disaster_type}"
                ))

        return entities

    def _extract_location_entities(self, text: str) -> List[Entity]:
        """Extract location entities"""
        entities = []

        for pattern in self.location_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                entities.append(Entity(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label="LOCATION"
                ))

        return entities

    def _extract_time_entities(self, text: str) -> List[Entity]:
        """Extract time entities"""
        entities = []

        for pattern in self.time_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                entities.append(Entity(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label="TIME"
                ))

        return entities

    def _extract_damage_entities(self, text: str) -> List[Entity]:
        """Extract damage entities"""
        entities = []

        for pattern in self.damage_patterns:
            for match in re.finditer(pattern, text):
                # Extend match to include surrounding context
                start, end = match.span()
                extended_start = max(0, start - 20)
                extended_end = min(len(text), end + 20)

                # Find sentence boundaries
                sentence_start = text.rfind('.', 0, start) + 1 if text.rfind('.', 0, start) != -1 else 0
                sentence_end = text.find('.', end)
                if sentence_end == -1:
                    sentence_end = len(text)

                entities.append(Entity(
                    start=sentence_start,
                    end=sentence_end,
                    text=text[sentence_start:sentence_end].strip(),
                    label="DAMAGE"
                ))

        return entities

    def _extract_response_entities(self, text: str) -> List[Entity]:
        """Extract response entities"""
        entities = []

        for pattern in self.response_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Extend to phrase
                word_start = text.rfind(' ', 0, start) + 1 if text.rfind(' ', 0, start) != -1 else 0
                word_end = text.find(' ', end)
                if word_end == -1:
                    word_end = len(text)

                entities.append(Entity(
                    start=word_start,
                    end=word_end,
                    text=text[word_start:word_end].strip(),
                    label="RESPONSE"
                ))

        return entities

    def _extract_impact_entities(self, text: str) -> List[Entity]:
        """Extract impact entities"""
        entities = []

        for pattern in self.impact_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Extend to sentence
                sentence_start = text.rfind('.', 0, start) + 1 if text.rfind('.', 0, start) != -1 else 0
                sentence_end = text.find('.', end)
                if sentence_end == -1:
                    sentence_end = len(text)

                entities.append(Entity(
                    start=sentence_start,
                    end=sentence_end,
                    text=text[sentence_start:sentence_end].strip(),
                    label="IMPACT"
                ))

        return entities

    def _extract_forecast_entities(self, text: str) -> List[Entity]:
        """Extract forecast entities"""
        entities = []

        for pattern in self.forecast_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Extend to sentence
                sentence_start = text.rfind('.', 0, start) + 1 if text.rfind('.', 0, start) != -1 else 0
                sentence_end = text.find('.', end)
                if sentence_end == -1:
                    sentence_end = len(text)

                entities.append(Entity(
                    start=sentence_start,
                    end=sentence_end,
                    text=text[sentence_start:sentence_end].strip(),
                    label="FORECAST"
                ))

        return entities

    def _resolve_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities by keeping longer ones"""
        if not entities:
            return entities

        # Sort by start position, then by length (descending)
        entities.sort(key=lambda x: (x.start, -(x.end - x.start)))

        resolved = []
        for entity in entities:
            # Check if this entity overlaps with any in resolved
            overlaps = False
            for existing in resolved:
                if (entity.start < existing.end and entity.end > existing.start):
                    overlaps = True
                    break

            if not overlaps:
                resolved.append(entity)

        return resolved

    def _extract_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        relations = []

        # Simple rule-based relation extraction
        disaster_entities = [e for e in entities if e.label.startswith("DISASTER")]
        location_entities = [e for e in entities if e.label == "LOCATION"]
        time_entities = [e for e in entities if e.label == "TIME"]
        damage_entities = [e for e in entities if e.label == "DAMAGE"]
        impact_entities = [e for e in entities if e.label == "IMPACT"]

        # Disaster -> Location relations
        for disaster in disaster_entities:
            for location in location_entities:
                if abs(disaster.start - location.start) < 200:  # Within 200 chars
                    relations.append(Relation(
                        head=disaster,
                        tail=location,
                        relation_type="LOCATION_OF"
                    ))

        # Disaster -> Time relations
        for disaster in disaster_entities:
            for time in time_entities:
                if abs(disaster.start - time.start) < 150:
                    relations.append(Relation(
                        head=disaster,
                        tail=time,
                        relation_type="TIME_OF"
                    ))

        # Disaster -> Damage relations
        for disaster in disaster_entities:
            for damage in damage_entities:
                if abs(disaster.start - damage.start) < 300:
                    relations.append(Relation(
                        head=disaster,
                        tail=damage,
                        relation_type="CAUSE_OF"
                    ))

        # Disaster -> Impact relations
        for disaster in disaster_entities:
            for impact in impact_entities:
                if abs(disaster.start - impact.start) < 300:
                    relations.append(Relation(
                        head=disaster,
                        tail=impact,
                        relation_type="IMPACT_OF"
                    ))

        return relations

    def _classify_event_type(self, text: str) -> Optional[str]:
        """Classify the type of disaster event"""
        text_lower = text.lower()

        for keyword, disaster_type in self.disaster_keywords.items():
            if keyword in text_lower:
                return disaster_type

        return None


def convert_to_conll_format(documents: List[AnnotatedDocument]) -> str:
    """Convert annotated documents to CoNLL format for NER training"""
    conll_lines = []

    for doc in documents:
        tokens = []
        labels = []

        # Simple tokenization (can be improved with proper tokenizer)
        words = doc.text.split()
        word_positions = []
        pos = 0

        for word in words:
            start_pos = doc.text.find(word, pos)
            end_pos = start_pos + len(word)
            word_positions.append((start_pos, end_pos))
            pos = end_pos

        # Assign labels
        current_labels = ["O"] * len(words)

        for entity in doc.entities:
            # Find tokens that overlap with entity
            for i, (start_pos, end_pos) in enumerate(word_positions):
                if (start_pos >= entity.start and start_pos < entity.end) or \
                   (end_pos > entity.start and end_pos <= entity.end) or \
                   (start_pos <= entity.start and end_pos >= entity.end):

                    if current_labels[i] == "O":
                        if start_pos == entity.start:
                            current_labels[i] = f"B-{entity.label}"
                        else:
                            current_labels[i] = f"I-{entity.label}"

        # Create CoNLL lines
        for i, (word, label) in enumerate(zip(words, current_labels)):
            conll_lines.append(f"{word}\t{label}")
        conll_lines.append("")  # Empty line between sentences

    return "\n".join(conll_lines)


def convert_to_json_format(documents: List[AnnotatedDocument]) -> List[Dict[str, Any]]:
    """Convert to JSON format suitable for training"""
    json_data = []

    for doc in documents:
        json_doc = {
            "id": doc.id,
            "text": doc.text,
            "event_type": doc.event_type,
            "entities": [
                {
                    "start": entity.start,
                    "end": entity.end,
                    "text": entity.text,
                    "label": entity.label,
                    "confidence": entity.confidence
                }
                for entity in doc.entities
            ],
            "relations": [
                {
                    "head": {
                        "start": rel.head.start,
                        "end": rel.head.end,
                        "text": rel.head.text,
                        "label": rel.head.label
                    },
                    "tail": {
                        "start": rel.tail.start,
                        "end": rel.tail.end,
                        "text": rel.tail.text,
                        "label": rel.tail.label
                    },
                    "relation_type": rel.relation_type,
                    "confidence": rel.confidence
                }
                for rel in doc.relations
            ] if doc.relations else [],
            "metadata": doc.metadata
        }
        json_data.append(json_doc)

    return json_data


def main():
    """Main annotation function"""
    parser = argparse.ArgumentParser(description="Disaster Information Annotation Tool")
    parser.add_argument("--input", required=True, help="Input file (JSON or text)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--format", choices=["json", "conll"], default="json",
                       help="Output format")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load input data
    if args.input.endswith(".json"):
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        if isinstance(input_data, list):
            texts = [item.get("content", item.get("text", "")) for item in input_data]
            ids = [item.get("id", f"doc_{i}") for i, item in enumerate(input_data)]
        else:
            texts = [input_data.get("content", input_data.get("text", ""))]
            ids = [input_data.get("id", "doc_0")]
    else:
        # Plain text file
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
            texts = content.split("\n\n")  # Split by double newlines
            ids = [f"doc_{i}" for i in range(len(texts))]

    # Initialize annotator
    annotator = DisasterAnnotator()

    # Process documents
    annotated_docs = []
    for text, doc_id in zip(texts, ids):
        if text.strip():  # Skip empty texts
            try:
                annotated_doc = annotator.annotate_text(text.strip(), doc_id)
                annotated_docs.append(annotated_doc)
                print(f"✅ Annotated document: {doc_id}")
            except Exception as e:
                print(f"❌ Error annotating {doc_id}: {str(e)}")

    # Save results
    if args.format == "json":
        json_data = convert_to_json_format(annotated_docs)
        output_file = os.path.join(args.output, "annotated_data.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(annotated_docs)} annotated documents to {output_file}")

    elif args.format == "conll":
        conll_data = convert_to_conll_format(annotated_docs)
        output_file = os.path.join(args.output, "annotated_data.conll")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(conll_data)

        print(f"💾 Saved CoNLL format to {output_file}")

    # Save statistics
    stats = {
        "total_documents": len(annotated_docs),
        "total_entities": sum(len(doc.entities) for doc in annotated_docs),
        "total_relations": sum(len(doc.relations) for doc in annotated_docs),
        "entity_types": {},
        "relation_types": {},
        "event_types": {}
    }

    for doc in annotated_docs:
        for entity in doc.entities:
            label = entity.label.split("-")[0]  # Remove B-/I- prefix
            stats["entity_types"][label] = stats["entity_types"].get(label, 0) + 1

        for relation in doc.relations:
            stats["relation_types"][relation.relation_type] = \
                stats["relation_types"].get(relation.relation_type, 0) + 1

        if doc.event_type:
            stats["event_types"][doc.event_type] = \
                stats["event_types"].get(doc.event_type, 0) + 1

    stats_file = os.path.join(args.output, "annotation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"📊 Statistics saved to {stats_file}")
    print(f"📈 Total entities: {stats['total_entities']}")
    print(f"🔗 Total relations: {stats['total_relations']}")


if __name__ == "__main__":
    main()