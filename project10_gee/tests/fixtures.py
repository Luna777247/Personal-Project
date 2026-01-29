#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock fixtures and test data for flood detection tests.

Provides reusable mock objects, sample geometries,
and test datasets for unit and integration testing.
"""

import ee
from unittest.mock import Mock, MagicMock
from typing import Dict, Any


class MockEEImage:
    """Mock Earth Engine Image for testing."""
    
    def __init__(self, constant_value=0):
        """
        Initialize mock image with constant value.
        
        Args:
            constant_value: Constant pixel value
        """
        self.value = constant_value
        self.bands_data = {'constant': constant_value}
    
    def lt(self, threshold):
        """Mock less than operation."""
        return MockEEImage(1 if self.value < threshold else 0)
    
    def gt(self, threshold):
        """Mock greater than operation."""
        return MockEEImage(1 if self.value > threshold else 0)
    
    def And(self, other):
        """Mock AND operation."""
        result = self.value and (other.value if hasattr(other, 'value') else other)
        return MockEEImage(1 if result else 0)
    
    def Or(self, other):
        """Mock OR operation."""
        result = self.value or (other.value if hasattr(other, 'value') else other)
        return MockEEImage(1 if result else 0)
    
    def Not(self):
        """Mock NOT operation."""
        return MockEEImage(0 if self.value else 1)
    
    def clip(self, geometry):
        """Mock clip operation."""
        return self
    
    def select(self, bands):
        """Mock select bands."""
        return self
    
    def reduceRegion(self, **kwargs):
        """Mock reduce region."""
        mock_result = Mock()
        mock_result.getInfo = Mock(return_value=self.bands_data)
        mock_result.get = Mock(return_value=self.value)
        return mock_result
    
    def getInfo(self):
        """Mock getInfo."""
        return self.bands_data


class MockEEGeometry:
    """Mock Earth Engine Geometry for testing."""
    
    def __init__(self, area_m2=1e6):
        """
        Initialize mock geometry.
        
        Args:
            area_m2: Area in square meters (default 1 km²)
        """
        self._area_m2 = area_m2
    
    def area(self, max_error=1):
        """Mock area calculation."""
        mock_result = Mock()
        mock_result.getInfo = Mock(return_value=self._area_m2)
        return mock_result
    
    def bounds(self):
        """Mock bounds."""
        return self
    
    def intersection(self, other, max_error=1):
        """Mock intersection."""
        return self


class MockEEImageCollection:
    """Mock Earth Engine ImageCollection for testing."""
    
    def __init__(self, images=None, count=0):
        """
        Initialize mock collection.
        
        Args:
            images: List of mock images
            count: Number of images in collection
        """
        self.images = images or []
        self._count = count
    
    def filterBounds(self, geometry):
        """Mock filter by bounds."""
        return self
    
    def filterDate(self, start, end):
        """Mock filter by date."""
        return self
    
    def filter(self, filter_obj):
        """Mock generic filter."""
        return self
    
    def select(self, bands):
        """Mock select bands."""
        return self
    
    def size(self):
        """Mock size."""
        mock_result = Mock()
        mock_result.getInfo = Mock(return_value=self._count)
        return mock_result
    
    def first(self):
        """Mock first image."""
        return self.images[0] if self.images else MockEEImage()
    
    def aggregate_array(self, property_name):
        """Mock aggregate array."""
        mock_result = Mock()
        mock_result.getInfo = Mock(return_value=[])
        return mock_result
    
    def toList(self, count):
        """Mock to list."""
        return self.images


# Test geometries
TEST_GEOMETRIES = {
    'small_area': {
        'coords': [107.5, 16.3, 107.51, 16.31],
        'area_km2': 1.0,
        'description': 'Small test area ~1 km²'
    },
    'medium_area': {
        'coords': [107.5, 16.3, 107.6, 16.4],
        'area_km2': 100.0,
        'description': 'Medium test area ~100 km²'
    },
    'large_area': {
        'coords': [107.0, 16.0, 108.0, 17.0],
        'area_km2': 10000.0,
        'description': 'Large test area ~10,000 km²'
    },
    'boundary_area': {
        'coords': [107.5, 16.3, 107.6, 16.4],
        'area_km2': 1000.0,
        'description': 'Boundary area exactly 1000 km²'
    }
}

# Test SAR backscatter values (in dB)
TEST_VV_VALUES = {
    'water': -22.0,      # Typical water backscatter
    'urban': -12.0,      # Urban area
    'vegetation': -15.0, # Vegetated area
    'bare_soil': -10.0,  # Bare soil
    'threshold_ems': -18.0,  # EMS Conservative threshold
}

# Test dates
TEST_DATES = {
    'event': '2025-09-15',
    'baseline_start': '2025-07-15',
    'baseline_end': '2025-09-14',
    'validation_start': '2025-09-08',
    'validation_end': '2025-09-22'
}

# Test thresholds
TEST_THRESHOLDS = {
    'slope': 15.0,       # degrees
    'hand': 20.0,        # meters
    'cloud_cover': 40.0, # percentage
    'permanent_water': 95.0  # occurrence percentage
}


def create_test_geometry(geometry_type='small_area'):
    """
    Create a test geometry using Earth Engine.
    
    Args:
        geometry_type: Key from TEST_GEOMETRIES
    
    Returns:
        ee.Geometry: Test geometry
    """
    geom_data = TEST_GEOMETRIES[geometry_type]
    coords = geom_data['coords']
    return ee.Geometry.Rectangle(coords)


def create_mock_config():
    """
    Create a mock Config object for testing.
    
    Returns:
        Mock: Mock configuration object
    """
    mock_config = Mock()
    mock_config.country = "Viet Nam"
    mock_config.provinces = ["Test Province"]
    mock_config.start_date = TEST_DATES['baseline_start']
    mock_config.end_date = TEST_DATES['event']
    mock_config.ems_threshold = TEST_THRESHOLDS['slope']
    mock_config.adaptive_k = 1.0
    mock_config.baseline_days = 60
    mock_config.baseline_vv_threshold = -8.5
    mock_config.hand_threshold = TEST_THRESHOLDS['hand']
    mock_config.slope_threshold = TEST_THRESHOLDS['slope']
    mock_config.kernel_size = 1
    mock_config.confidence_thresholds = [80, 60, 40, 20]
    mock_config.ensemble_vote_threshold = 3
    mock_config.max_cloud_cover = TEST_THRESHOLDS['cloud_cover']
    mock_config.max_pixels = 1e9
    mock_config.scale_small_area = 10
    mock_config.scale_large_area = 30
    mock_config.max_districts = 100
    mock_config.optical_search_days = 7
    
    def mock_get_adaptive_scale(area_km2):
        return 10 if area_km2 < 1000 else 30
    
    mock_config.get_adaptive_scale = mock_get_adaptive_scale
    
    return mock_config


def create_synthetic_sar_image(vv_value=-15.0, geometry=None):
    """
    Create synthetic SAR image for testing.
    
    Args:
        vv_value: VV backscatter value in dB
        geometry: Optional geometry to clip
    
    Returns:
        ee.Image: Synthetic SAR image
    """
    image = ee.Image.constant(vv_value)
    
    if geometry:
        image = image.clip(geometry)
    
    return image


def create_test_masks(geometry):
    """
    Create test masks for flood detection.
    
    Args:
        geometry: Test geometry
    
    Returns:
        Dict[str, ee.Image]: Dictionary of test masks
    """
    return {
        'slope_mask': ee.Image(1).clip(geometry),
        'permanent_water_mask': ee.Image(1).clip(geometry),
        'hand_mask': ee.Image(1).clip(geometry),
        'combined_mask': ee.Image(1).clip(geometry)
    }


def get_test_statistics():
    """
    Get sample statistics for testing.
    
    Returns:
        Dict[str, float]: Test statistics
    """
    return {
        'VV_mean': -15.5,
        'VV_stdDev': 3.2,
        'VV_min': -25.0,
        'VV_max': -8.0,
        'area_km2': 100.0,
        'water_percentage': 15.5
    }


def create_validation_data():
    """
    Create validation data for testing metrics.
    
    Returns:
        Dict[str, Any]: Validation data including confusion matrix values
    """
    return {
        'true_positives': 800,
        'false_positives': 200,
        'true_negatives': 7000,
        'false_negatives': 100,
        'total_pixels': 8100,
        'precision': 0.8,
        'recall': 0.888,
        'f1_score': 0.842,
        'iou': 0.727
    }


# Sample configuration for testing
SAMPLE_CONFIG = {
    'country': 'Viet Nam',
    'provinces': ['Thua Thien - Hue', 'Da Nang City'],
    'start_date': '2025-09-01',
    'end_date': '2025-11-30',
    'ems_threshold': -18.0,
    'adaptive_k': 1.0,
    'baseline_days': 60,
    'scale_small': 10,
    'scale_large': 30
}


if __name__ == '__main__':
    print("Test Fixtures Module")
    print("="*50)
    print("\nAvailable Test Geometries:")
    for key, value in TEST_GEOMETRIES.items():
        print(f"  {key}: {value['description']}")
    
    print("\nTest VV Values (dB):")
    for key, value in TEST_VV_VALUES.items():
        print(f"  {key}: {value}")
    
    print("\nTest Dates:")
    for key, value in TEST_DATES.items():
        print(f"  {key}: {value}")
