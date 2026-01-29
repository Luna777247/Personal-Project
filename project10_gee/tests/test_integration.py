#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for flood detection methods.

Tests the entire flood detection pipeline including:
- Data loading
- Preprocessing
- Multiple detection algorithms
- Ensemble voting
- Validation
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import ee

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFloodDetectionPipeline(unittest.TestCase):
    """Integration tests for flood detection workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine once for all tests."""
        try:
            ee.Initialize(project='driven-torus-431807-u3')
            cls.ee_available = True
        except:
            cls.ee_available = False
            print("WARNING: Earth Engine not available. Integration tests will be skipped.")
    
    def setUp(self):
        """Set up test geometry and basic parameters."""
        if not self.ee_available:
            self.skipTest("Earth Engine not available")
        
        # Small test area (reduce computation time)
        self.test_geom = ee.Geometry.Rectangle([107.5, 16.3, 107.6, 16.4])
        self.test_date_start = "2025-09-01"
        self.test_date_end = "2025-09-10"
    
    def test_data_loading_s1(self):
        """Test Sentinel-1 data loading."""
        from gee_khoanh_cùng_ngập_lụt_v2 import load_s1
        
        col, count, ids = load_s1("A", self.test_geom, 
                                   self.test_date_start, self.test_date_end)
        
        self.assertIsInstance(col, ee.ImageCollection)
        self.assertIsInstance(count, int)
        self.assertIsInstance(ids, list)
        self.assertGreaterEqual(count, 0)
    
    def test_data_loading_s2(self):
        """Test Sentinel-2 data loading."""
        from gee_khoanh_cùng_ngập_lụt_v2 import load_s2
        
        col, count, ids = load_s2(self.test_geom, 
                                   self.test_date_start, self.test_date_end,
                                   cloud=50)
        
        self.assertIsInstance(col, ee.ImageCollection)
        self.assertIsInstance(count, int)
        self.assertIsInstance(ids, list)
        self.assertGreaterEqual(count, 0)
    
    def test_dem_slope_calculation(self):
        """Test DEM loading and slope calculation."""
        dem = ee.Image("USGS/SRTMGL1_003").clip(self.test_geom)
        slope = ee.Terrain.slope(dem)
        
        # Verify images are valid
        self.assertIsInstance(dem, ee.Image)
        self.assertIsInstance(slope, ee.Image)
        
        # Check that slope values are reasonable (0-90 degrees)
        slope_stats = slope.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_geom,
            scale=30,
            maxPixels=1e8
        ).getInfo()
        
        self.assertIn('slope_min', slope_stats)
        self.assertIn('slope_max', slope_stats)
        self.assertGreaterEqual(slope_stats['slope_min'], 0)
        self.assertLessEqual(slope_stats['slope_max'], 90)
    
    def test_permanent_water_loading(self):
        """Test permanent water data loading."""
        jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
        permanent_water = jrc.select("occurrence").gt(95).clip(self.test_geom)
        
        self.assertIsInstance(permanent_water, ee.Image)
        
        # Check that values are binary (0 or 1)
        stats = permanent_water.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_geom,
            scale=30,
            maxPixels=1e8
        ).getInfo()
        
        # Binary mask should only have 0 and 1
        self.assertIn(stats.get('occurrence_min', 0), [0, 1])
        self.assertIn(stats.get('occurrence_max', 0), [0, 1])
    
    def test_mask_creation(self):
        """Test combined mask creation."""
        dem = ee.Image("USGS/SRTMGL1_003").clip(self.test_geom)
        slope = ee.Terrain.slope(dem)
        
        jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
        permanent_water = jrc.select("occurrence").gt(95).clip(self.test_geom)
        
        merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
        hand = merit_hydro.select('hnd').clip(self.test_geom)
        
        # Create masks
        slope_mask = slope.lt(15)
        permanent_water_mask = permanent_water.Not()
        hand_mask = hand.lt(20)
        
        # Combine
        combined_mask = slope_mask.And(permanent_water_mask).And(hand_mask)
        
        self.assertIsInstance(combined_mask, ee.Image)
        
        # Check that combined mask is binary
        stats = combined_mask.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_geom,
            scale=30,
            maxPixels=1e8
        ).getInfo()
        
        # Should be binary (0 or 1)
        values = list(stats.values())
        if values:  # If not empty
            self.assertTrue(all(v in [0, 1] for v in values if v is not None))
    
    def test_vv_statistics_calculation(self):
        """Test VV backscatter statistics calculation."""
        from gee_khoanh_cùng_ngập_lụt_v2 import load_s1
        
        col, count, ids = load_s1("A", self.test_geom, 
                                   self.test_date_start, self.test_date_end)
        
        if count == 0:
            self.skipTest("No Sentinel-1 images available for test period")
        
        # Get first image
        image = ee.Image(col.first())
        
        # Calculate statistics
        stats = image.select('VV').reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
            geometry=self.test_geom,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        
        self.assertIn('VV_mean', stats)
        self.assertIn('VV_stdDev', stats)
        
        # VV should be in reasonable dB range
        if stats['VV_mean'] is not None:
            # Could be linear or dB
            mean_val = stats['VV_mean']
            # If in dB, should be negative
            if mean_val < 0:
                self.assertGreater(mean_val, -30, "VV mean should be > -30 dB")
                self.assertLess(mean_val, 10, "VV mean should be < 10 dB")


class TestFloodDetectionMethods(unittest.TestCase):
    """Test individual flood detection methods."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine once for all tests."""
        try:
            ee.Initialize(project='driven-torus-431807-u3')
            cls.ee_available = True
        except:
            cls.ee_available = False
    
    def setUp(self):
        """Set up test data."""
        if not self.ee_available:
            self.skipTest("Earth Engine not available")
        
        self.test_geom = ee.Geometry.Rectangle([107.5, 16.3, 107.55, 16.35])
        
        # Create synthetic test image in dB
        self.test_image = ee.Image.constant(-15.0).clip(self.test_geom)
    
    def test_ems_method(self):
        """Test EMS Conservative method."""
        threshold = -18.0
        water_mask = self.test_image.lt(threshold)
        
        self.assertIsInstance(water_mask, ee.Image)
        
        # Since test image is -15 dB and threshold is -18 dB,
        # no water should be detected
        stats = water_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=self.test_geom,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        
        # Should be 0 (no water detected)
        sum_val = list(stats.values())[0] if stats.values() else 0
        self.assertEqual(sum_val, 0, "No water should be detected with -15 dB image")
        
        # Test with lower threshold
        threshold_low = -10.0
        water_mask_low = self.test_image.lt(threshold_low)
        stats_low = water_mask_low.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=self.test_geom,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        
        # Should detect water
        sum_val_low = list(stats_low.values())[0] if stats_low.values() else 0
        self.assertGreater(sum_val_low, 0, "Water should be detected with -10 dB threshold")
    
    def test_adaptive_landcover_method(self):
        """Test Adaptive Landcover method."""
        worldcover = ee.Image("ESA/WorldCover/v200/2021").clip(self.test_geom)
        
        self.assertIsInstance(worldcover, ee.Image)
        
        # Create masks for different land types
        urban = worldcover.eq(50)
        rural = worldcover.eq(40).Or(worldcover.eq(30))
        
        # Test thresholds
        urban_water = self.test_image.lt(-14).And(urban)
        rural_water = self.test_image.lt(-17).And(rural)
        
        self.assertIsInstance(urban_water, ee.Image)
        self.assertIsInstance(rural_water, ee.Image)
    
    def test_mask_combination(self):
        """Test mask combination logic."""
        mask1 = ee.Image(1)
        mask2 = ee.Image(1)
        mask3 = ee.Image(0)
        
        # Test AND operation
        combined_and = mask1.And(mask2).And(mask3)
        result = combined_and.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=self.test_geom,
            scale=10
        ).getInfo()
        
        self.assertEqual(list(result.values())[0], 0, "AND with 0 should be 0")
        
        # Test OR operation
        combined_or = mask1.Or(mask2).Or(mask3)
        result_or = combined_or.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=self.test_geom,
            scale=10
        ).getInfo()
        
        self.assertEqual(list(result_or.values())[0], 1, "OR with 1 should be 1")


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing steps."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine."""
        try:
            ee.Initialize(project='driven-torus-431807-u3')
            cls.ee_available = True
        except:
            cls.ee_available = False
    
    def setUp(self):
        """Set up test data."""
        if not self.ee_available:
            self.skipTest("Earth Engine not available")
        
        self.test_geom = ee.Geometry.Point([107.5, 16.3]).buffer(1000)
        self.test_image = ee.Image.constant(100).clip(self.test_geom)
    
    def test_db_conversion(self):
        """Test linear to dB conversion."""
        # Convert linear to dB: 10 * log10(value)
        db_image = self.test_image.log10().multiply(10)
        
        # 10 * log10(100) = 10 * 2 = 20 dB
        result = db_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=self.test_geom,
            scale=10
        ).getInfo()
        
        db_value = list(result.values())[0]
        self.assertAlmostEqual(db_value, 20.0, places=1)
    
    def test_clamp_operation(self):
        """Test clamp operation."""
        # Create image with range -40 to +10 dB
        image = ee.Image.constant(-40).addBands(ee.Image.constant(10))
        
        # Clamp to [-30, 0]
        clamped = image.clamp(-30, 0)
        
        stats = clamped.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.test_geom,
            scale=10
        ).getInfo()
        
        # All values should be within [-30, 0]
        for key, value in stats.items():
            if value is not None:
                self.assertGreaterEqual(value, -30)
                self.assertLessEqual(value, 0)
    
    def test_speckle_filter(self):
        """Test speckle filtering with focal_median."""
        # Apply focal median filter
        filtered = self.test_image.focal_median(2.5, 'circle', 'meters')
        
        self.assertIsInstance(filtered, ee.Image)
        
        # Filtered image should have similar mean but lower std dev
        original_stats = self.test_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
            geometry=self.test_geom,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        
        filtered_stats = filtered.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
            geometry=self.test_geom,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        
        # Mean should be similar (constant image)
        if original_stats and filtered_stats:
            orig_mean = [v for k, v in original_stats.items() if 'mean' in k][0]
            filt_mean = [v for k, v in filtered_stats.items() if 'mean' in k][0]
            
            if orig_mean and filt_mean:
                self.assertAlmostEqual(orig_mean, filt_mean, delta=1.0)


class TestValidation(unittest.TestCase):
    """Test validation and metrics calculation."""
    
    def test_iou_calculation(self):
        """Test IoU calculation logic."""
        # Mock scenario
        intersection_area = 100
        union_area = 200
        
        iou = intersection_area / union_area
        self.assertEqual(iou, 0.5)
    
    def test_precision_recall_calculation(self):
        """Test precision and recall calculation."""
        tp = 80  # True positives
        fp = 20  # False positives
        fn = 10  # False negatives
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        self.assertEqual(precision, 0.8)
        self.assertAlmostEqual(recall, 0.888, places=3)
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        precision = 0.8
        recall = 0.9
        
        f1 = 2 * (precision * recall) / (precision + recall)
        expected = 2 * (0.8 * 0.9) / (0.8 + 0.9)
        
        self.assertAlmostEqual(f1, expected, places=3)
        self.assertAlmostEqual(f1, 0.847, places=3)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
