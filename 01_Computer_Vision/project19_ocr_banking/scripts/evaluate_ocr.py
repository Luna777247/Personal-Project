"""
Evaluation script for OCR accuracy
"""
import json
from pathlib import Path
import argparse
import sys
from typing import List, Dict
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline import OCRPipeline


def load_ground_truth(json_path: str) -> Dict:
    """Load ground truth annotations"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def evaluate_ocr(pipeline: OCRPipeline, test_data: Dict) -> Dict:
    """
    Evaluate OCR accuracy
    
    Args:
        pipeline: OCR pipeline
        test_data: Ground truth data
        
    Returns:
        Evaluation results
    """
    results = {
        'total': len(test_data['images']),
        'correct_document_type': 0,
        'text_similarity': [],
        'field_accuracy': {},
        'per_image': []
    }
    
    for item in test_data['images']:
        image_path = item['path']
        ground_truth = item['ground_truth']
        
        print(f"Processing: {image_path}")
        
        try:
            # Run OCR
            ocr_result = pipeline.process_image(image_path)
            
            # Evaluate document type
            if ocr_result['document_type'] == ground_truth.get('document_type'):
                results['correct_document_type'] += 1
            
            # Evaluate text similarity
            text_sim = calculate_similarity(
                ocr_result['raw_text'],
                ground_truth.get('text', '')
            )
            results['text_similarity'].append(text_sim)
            
            # Evaluate field accuracy
            gt_fields = ground_truth.get('fields', {})
            ocr_fields = ocr_result['extracted_fields']
            
            image_results = {
                'image': image_path,
                'document_type_correct': ocr_result['document_type'] == ground_truth.get('document_type'),
                'text_similarity': text_sim,
                'field_matches': {}
            }
            
            for field_name, gt_value in gt_fields.items():
                ocr_value = ocr_fields.get(field_name)
                
                if field_name not in results['field_accuracy']:
                    results['field_accuracy'][field_name] = {'correct': 0, 'total': 0}
                
                results['field_accuracy'][field_name]['total'] += 1
                
                if gt_value and ocr_value:
                    similarity = calculate_similarity(str(gt_value), str(ocr_value))
                    image_results['field_matches'][field_name] = similarity
                    
                    if similarity > 0.9:  # Consider correct if > 90% similar
                        results['field_accuracy'][field_name]['correct'] += 1
            
            results['per_image'].append(image_results)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Calculate averages
    results['document_type_accuracy'] = results['correct_document_type'] / results['total']
    results['avg_text_similarity'] = sum(results['text_similarity']) / len(results['text_similarity'])
    
    for field_name, stats in results['field_accuracy'].items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return results


def print_evaluation_results(results: Dict):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("OCR EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTotal Images: {results['total']}")
    print(f"Document Type Accuracy: {results['document_type_accuracy']:.2%}")
    print(f"Average Text Similarity: {results['avg_text_similarity']:.2%}")
    
    print("\nField-wise Accuracy:")
    print("-"*60)
    for field_name, stats in results['field_accuracy'].items():
        print(f"  {field_name:20s}: {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:.2%})")
    
    print("\nPer-Image Results:")
    print("-"*60)
    for item in results['per_image']:
        print(f"\n{item['image']}")
        print(f"  Document Type: {'✓' if item['document_type_correct'] else '✗'}")
        print(f"  Text Similarity: {item['text_similarity']:.2%}")
        print(f"  Field Matches:")
        for field, sim in item['field_matches'].items():
            print(f"    {field}: {sim:.2%}")
    
    print("\n" + "="*60)


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy")
    parser.add_argument('ground_truth', help='Path to ground truth JSON file')
    parser.add_argument('-o', '--output', default='evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('-d', '--detector', default='craft',
                       choices=['craft', 'dbnet'],
                       help='Detector type')
    parser.add_argument('-r', '--recognizer', default='vietocr',
                       choices=['vietocr', 'paddleocr', 'easyocr'],
                       help='Recognizer type')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"Loading ground truth from: {args.ground_truth}")
    test_data = load_ground_truth(args.ground_truth)
    
    # Initialize pipeline
    print(f"Initializing OCR pipeline: {args.detector} + {args.recognizer}")
    pipeline = OCRPipeline(
        detector_type=args.detector,
        recognizer_type=args.recognizer,
        device=args.device,
        use_postprocessing=True
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_ocr(pipeline, test_data)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    save_results(results, args.output)
