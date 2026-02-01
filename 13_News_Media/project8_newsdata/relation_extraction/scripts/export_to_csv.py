#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Relation Extraction Results to CSV
"""

import json
import csv
import os
from typing import List, Dict, Any

def export_results_to_csv():
    """Export relation extraction results to CSV"""
    data_dir = 'data'
    output_file = os.path.join(data_dir, 'relation_extraction_results.csv')

    all_results = []

    # Load rule-based results
    rule_file = os.path.join(data_dir, 're_results_rule.json')
    if os.path.exists(rule_file):
        with open(rule_file, 'r', encoding='utf-8') as f:
            rule_results = json.load(f)
        all_results.extend(rule_results)

    # Load LLM results
    llm_file = os.path.join(data_dir, 're_results_llm.json')
    if os.path.exists(llm_file):
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_results = json.load(f)
        all_results.extend(llm_results)

    if not all_results:
        print("No results found to export")
        return

    # Write to CSV
    fieldnames = ['article_title', 'article_url', 'article_source', 'relations', 'processing_time', 'model_used', 'confidence_score', 'relation_count']

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Convert relations list to string
            result['relations'] = str(result.get('relations', []))
            writer.writerow(result)

    print(f"Results exported to {output_file}")
    print(f"Total records: {len(all_results)}")

if __name__ == "__main__":
    export_results_to_csv()