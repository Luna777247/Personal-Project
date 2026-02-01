"""
Complete OCR Pipeline
Combines detection, recognition, and extraction
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from detection.craft_detector import CRAFTDetector, DBNetDetector, create_detector
from recognition.vietocr_recognizer import VietOCRRecognizer, create_recognizer
from extraction.field_extractor import (
    FieldExtractor, DocumentTypeClassifier,
    CCCDExtractor, BankStatementExtractor
)
from extraction.postprocessing import TextPostProcessor, ConfidenceCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRPipeline:
    """End-to-end OCR pipeline for banking documents"""
    
    def __init__(self,
                 detector_type: str = 'craft',
                 recognizer_type: str = 'vietocr',
                 device: str = 'cpu',
                 use_postprocessing: bool = True):
        """
        Initialize OCR pipeline
        
        Args:
            detector_type: Text detector type ('craft', 'dbnet')
            recognizer_type: Text recognizer type ('vietocr', 'paddleocr', 'easyocr')
            device: Device to run on ('cpu', 'cuda')
            use_postprocessing: Apply post-processing
        """
        logger.info(f"Initializing OCR Pipeline: {detector_type} + {recognizer_type}")
        
        # Initialize detector
        self.detector = create_detector(
            detector_type,
            cuda=(device == 'cuda')
        )
        
        # Initialize recognizer
        recognizer_kwargs = {}
        if recognizer_type == 'vietocr':
            recognizer_kwargs = {'device': device}
        elif recognizer_type == 'paddleocr':
            recognizer_kwargs = {'use_gpu': (device == 'cuda')}
        elif recognizer_type == 'easyocr':
            recognizer_kwargs = {'gpu': (device == 'cuda')}
        
        self.recognizer = create_recognizer(recognizer_type, **recognizer_kwargs)
        
        # Initialize extractors
        self.field_extractor = FieldExtractor()
        self.doc_classifier = DocumentTypeClassifier()
        self.cccd_extractor = CCCDExtractor()
        self.bank_extractor = BankStatementExtractor()
        
        # Initialize post-processor
        self.use_postprocessing = use_postprocessing
        if use_postprocessing:
            self.postprocessor = TextPostProcessor()
            self.confidence_calc = ConfidenceCalculator()
        
        logger.info("OCR Pipeline initialized successfully")
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process single image through complete pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with OCR results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return self.process_image_array(image, image_path)
    
    def process_image_array(self, image: np.ndarray, source: str = "unknown") -> Dict:
        """
        Process image array through pipeline
        
        Args:
            image: Input image array
            source: Source identifier
            
        Returns:
            Dictionary with OCR results
        """
        result = {
            'source': source,
            'status': 'success',
            'document_type': 'unknown',
            'raw_text': '',
            'text_regions': [],
            'extracted_fields': {},
            'confidence': 0.0,
            'metadata': {}
        }
        
        try:
            # Step 1: Text Detection
            logger.info("Step 1: Detecting text regions...")
            boxes = self.detector.detect(image)
            logger.info(f"Detected {len(boxes)} text regions")
            
            if not boxes:
                result['status'] = 'no_text_detected'
                return result
            
            # Step 2: Text Recognition
            logger.info("Step 2: Recognizing text...")
            text_regions = []
            all_text = []
            
            for i, box in enumerate(boxes):
                # Crop text region
                crop = self._crop_box(image, box)
                
                if crop.size == 0:
                    continue
                
                # Recognize text
                text = self.recognizer.recognize(crop)
                
                # Post-process if enabled
                if self.use_postprocessing and text:
                    text = self.postprocessor.clean_text(text)
                
                text_regions.append({
                    'box': box.tolist() if isinstance(box, np.ndarray) else box,
                    'text': text,
                    'index': i
                })
                
                all_text.append(text)
            
            # Combine all text
            result['raw_text'] = ' '.join(all_text)
            result['text_regions'] = text_regions
            
            logger.info(f"Recognized {len(text_regions)} text regions")
            
            # Step 3: Document Classification
            logger.info("Step 3: Classifying document type...")
            doc_type = self.doc_classifier.classify(result['raw_text'])
            result['document_type'] = doc_type
            
            logger.info(f"Document type: {doc_type}")
            
            # Step 4: Field Extraction
            logger.info("Step 4: Extracting structured fields...")
            extracted_fields = self._extract_fields(result['raw_text'], doc_type)
            result['extracted_fields'] = extracted_fields
            
            # Step 5: Calculate Confidence
            if self.use_postprocessing:
                confidence = self._calculate_overall_confidence(
                    extracted_fields,
                    len(text_regions)
                )
                result['confidence'] = confidence
            
            logger.info(f"Processing completed with confidence: {result['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _crop_box(self, image: np.ndarray, box: np.ndarray, padding: int = 5) -> np.ndarray:
        """Crop text region from image"""
        h, w = image.shape[:2]
        
        # Get bounding rectangle
        if len(box) == 4:  # [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, box)
        elif len(box) == 8:  # Polygon
            xs = [box[0], box[2], box[4], box[6]]
            ys = [box[1], box[3], box[5], box[7]]
            x_min, y_min = int(min(xs)), int(min(ys))
            x_max, y_max = int(max(xs)), int(max(ys))
        else:
            return np.array([])
        
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop
        crop = image[y_min:y_max, x_min:x_max]
        
        return crop
    
    def _extract_fields(self, text: str, doc_type: str) -> Dict:
        """Extract structured fields based on document type"""
        if doc_type == 'cccd':
            return self.cccd_extractor.extract(text)
        elif doc_type == 'bank_statement':
            return self.bank_extractor.extract(text)
        else:
            # Generic extraction
            return {
                'dates': self.field_extractor.extract_multiple_values(text, 'date_dmy'),
                'money': self.field_extractor.extract_multiple_values(text, 'money_number'),
                'phones': self.field_extractor.extract_multiple_values(text, 'phone_vn')
            }
    
    def _calculate_overall_confidence(self, fields: Dict, num_regions: int) -> float:
        """Calculate overall confidence score"""
        if not fields:
            return 0.0
        
        # Calculate field confidences
        field_confidences = []
        
        for field_name, field_value in fields.items():
            if field_value:
                # Determine field type
                if 'id' in field_name.lower() or 'cccd' in field_name.lower():
                    field_type = 'cccd'
                elif 'date' in field_name.lower() or 'birth' in field_name.lower():
                    field_type = 'date_dmy'
                elif 'name' in field_name.lower():
                    field_type = 'vietnamese_name'
                else:
                    field_type = 'generic'
                
                if isinstance(field_value, str):
                    conf = self.confidence_calc.calculate_field_confidence(
                        field_value,
                        field_type
                    )
                    field_confidences.append(conf)
        
        # Calculate average
        if field_confidences:
            avg_confidence = sum(field_confidences) / len(field_confidences)
        else:
            avg_confidence = 0.5
        
        # Adjust based on number of regions detected
        region_factor = min(num_regions / 10, 1.0)
        
        overall_confidence = avg_confidence * 0.8 + region_factor * 0.2
        
        return overall_confidence
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'source': image_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize pipeline
    print("Initializing OCR pipeline...")
    pipeline = OCRPipeline(
        detector_type='craft',
        recognizer_type='vietocr',
        device='cpu',
        use_postprocessing=True
    )
    
    # Process image
    print(f"\nProcessing: {image_path}")
    result = pipeline.process_image(image_path)
    
    # Display results
    print("\n" + "="*50)
    print("OCR RESULTS")
    print("="*50)
    print(f"Status: {result['status']}")
    print(f"Document Type: {result['document_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nRaw Text:\n{result['raw_text']}")
    print(f"\nExtracted Fields:")
    for field, value in result['extracted_fields'].items():
        print(f"  {field}: {value}")
    print("="*50)
