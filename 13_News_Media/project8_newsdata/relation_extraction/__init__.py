#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Extraction Package
Trích xuất quan hệ giữa các entities trong bài báo thiên tai

Version: 1.0.0
Author: AI Assistant
Description: Advanced relation extraction for disaster information
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Relation Extraction for Disaster Information"

# Import main classes for easy access
from .scripts.relation_extractor import RelationExtractor, Relation
from .scripts.phobert_re_extractor import PhoBERTREExtractor
from .scripts.llm_re_extractor import LLMREExtractor
from .scripts.rule_based_re_extractor import RuleBasedREExtractor

__all__ = [
    'RelationExtractor',
    'Relation',
    'PhoBERTREExtractor',
    'LLMREExtractor',
    'RuleBasedREExtractor'
]