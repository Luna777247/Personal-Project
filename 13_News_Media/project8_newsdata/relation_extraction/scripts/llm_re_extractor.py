#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Relation Extraction
Sử dụng Large Language Models để trích xuất quan hệ
"""

import logging
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import os
from dotenv import load_dotenv

from .relation_extractor import RelationExtractor, Relation

logger = logging.getLogger(__name__)

class LLMREExtractor(RelationExtractor):
    """LLM-based relation extraction using prompts"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = "LLM-RE"
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 500)
        self.api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')

        # Load API key
        load_dotenv()
        self.api_key = os.getenv(self.api_key_env)

        # Prompt template
        self.prompt_template = config.get('prompt_template', self._get_default_prompt())

        # Fallback provider
        self.fallback_provider = config.get('fallback_provider')

        # Caching
        self.enable_caching = config.get('enable_caching', True)
        self.cache_dir = config.get('cache_dir', 'cache/re_cache')
        self.cache = {}

        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_cache()

        logger.info(f"Initialized LLM RE extractor with provider: {self.provider}")

    def _get_default_prompt(self) -> str:
        """Get default prompt template"""
        return """
Phân tích bài báo sau và trích xuất các quan hệ giữa các thực thể thiên tai.
Chỉ trả về JSON, không giải thích thêm.

Bài báo: {article_text}

Các thực thể đã được nhận diện: {entities}

Định dạng JSON:
{{
    "relations": [
        {{
            "head_entity": "thực thể đầu",
            "tail_entity": "thực thể cuối",
            "relation_type": "loại quan hệ",
            "confidence": 0.95
        }}
    ]
}}

Các loại quan hệ: {relation_types}
"""

    def _load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, 'llm_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk"""
        if not self.enable_caching:
            return

        cache_file = os.path.join(self.cache_dir, 'llm_cache.json')
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_cache_key(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Generate cache key"""
        entities_str = json.dumps(entities, sort_keys=True)
        return f"{hash(text + entities_str)}"

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        if not self.api_key:
            logger.error("OpenAI API key not found")
            return None

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.error("Groq API key not found")
            return None

        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': 'mixtral-8x7b-32768',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }

            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with fallback"""
        providers = [self.provider]
        if self.fallback_provider:
            providers.append(self.fallback_provider)

        for provider in providers:
            if provider == 'openai':
                response = self._call_openai(prompt)
            elif provider == 'groq':
                response = self._call_groq(prompt)
            else:
                logger.warning(f"Unknown provider: {provider}")
                continue

            if response:
                return response

        logger.error("All LLM providers failed")
        return None

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract relations"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in LLM response")
                return []

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            relations = data.get('relations', [])
            return relations

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Relation]:
        """
        Extract relations using LLM

        Args:
            text: Input text
            entities: List of entities from NER

        Returns:
            List of extracted relations
        """
        # Check cache first
        cache_key = self._get_cache_key(text, entities)
        if self.enable_caching and cache_key in self.cache:
            logger.info("Using cached LLM response")
            cached_relations = self.cache[cache_key]
            return [Relation(**r) for r in cached_relations]

        # Prepare prompt
        entities_text = "\n".join([f"- {e['text']} ({e.get('label', 'UNKNOWN')})" for e in entities])

        # Get relation types
        relation_types = list(self.relation_definitions.keys())

        prompt = self.prompt_template.format(
            article_text=text,
            entities=entities_text,
            relation_types=", ".join(relation_types)
        )

        # Call LLM
        logger.info("Calling LLM for relation extraction...")
        response = self._call_llm(prompt)

        if not response:
            logger.warning("LLM call failed")
            return []

        # Parse response
        relations_data = self._parse_llm_response(response)

        # Convert to Relation objects
        relations = []
        for r_data in relations_data:
            try:
                relation = Relation(
                    head_entity=r_data['head_entity'],
                    tail_entity=r_data['tail_entity'],
                    relation_type=r_data['relation_type'],
                    confidence=r_data.get('confidence', 0.8),
                    context=text,
                    sentence=""  # Could extract sentence containing entities
                )
                relations.append(relation)
            except KeyError as e:
                logger.warning(f"Missing key in relation data: {e}")
                continue

        # Cache results
        if self.enable_caching:
            self.cache[cache_key] = [
                {
                    'head_entity': r.head_entity,
                    'tail_entity': r.tail_entity,
                    'relation_type': r.relation_type,
                    'confidence': r.confidence,
                    'context': r.context,
                    'sentence': r.sentence
                } for r in relations
            ]
            self._save_cache()

        # Filter by confidence
        min_confidence = self.config.get('min_confidence', 0.5)
        relations = self.filter_relations_by_confidence(relations, min_confidence)

        logger.info(f"Extracted {len(relations)} relations using LLM")
        return relations