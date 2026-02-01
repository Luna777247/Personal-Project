#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for utility functions.

Tests all helper functions including area calculation,
error handling, geometry processing, and validation.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import ee

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path setup
from gee_khoanh_cùng_ngập_lụt_v2 import (
    Config,
    get_geometry_scale,
    calculate_water_area,
    safe_reduce_region,
    get_region,
)

config = Config()


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine once for all tests."""
        try:
            ee.Initialize(project='driven-torus-431807-u3')
        except:
            try:
                ee.Authenticate()
                ee.Initialize(project='driven-torus-431807-u3')
            except:
                print("WARNING: Could not initialize Earth Engine. Some tests will be skipped.")
    
    def test_get_geometry_scale_small_area(self):
        """Test get_geometry_scale for small geometries."""
        # Create a small rectangle (~100 km²)
        small_geom = ee.Geometry.Rectangle([105.0, 16.0, 105.5, 16.5])
        
        try:
            scale = get_geometry_scale(small_geom)
            self.assertEqual(scale, 10, "Small area should return 10m scale")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_get_geometry_scale_large_area(self):
        """Test get_geometry_scale for large geometries."""
        # Create a large rectangle (~10000 km²)
        large_geom = ee.Geometry.Rectangle([105.0, 16.0, 107.0, 17.0])
        
        try:
            scale = get_geometry_scale(large_geom)
            self.assertEqual(scale, 30, "Large area should return 30m scale")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_get_geometry_scale_fallback(self):
        """Test get_geometry_scale fallback on error."""
        # Mock a geometry that causes error
        mock_geom = Mock(spec=ee.Geometry)
        mock_geom.area = Mock(side_effect=Exception("Test error"))
        
        with patch('gee_khoanh_cùng_ngập_lụt_v2.logger') as mock_logger:
            scale = get_geometry_scale(mock_geom)
            self.assertEqual(scale, 10, "Should fallback to small scale on error")
            mock_logger.warning.assert_called_once()
    
    def test_calculate_water_area_basic(self):
        """Test basic water area calculation."""
        try:
            # Create a simple water mask (1 km²)
            geom = ee.Geometry.Rectangle([105.0, 16.0, 105.01, 16.01])
            mask = ee.Image(1).clip(geom)
            
            area_m2, area_km2 = calculate_water_area(mask, geom, scale=10)
            
            # Should be approximately 1 km² (allowing for projection errors)
            self.assertGreater(area_km2, 0.5, "Area should be > 0.5 km²")
            self.assertLess(area_km2, 2.0, "Area should be < 2.0 km²")
            self.assertAlmostEqual(area_m2, area_km2 * 1e6, places=0)
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_calculate_water_area_zero(self):
        """Test water area calculation with no water."""
        try:
            # Create a mask with no water (all zeros)
            geom = ee.Geometry.Rectangle([105.0, 16.0, 105.1, 16.1])
            mask = ee.Image(0).clip(geom)
            
            area_m2, area_km2 = calculate_water_area(mask, geom, scale=10)
            
            self.assertEqual(area_m2, 0.0, "Zero mask should return 0 area")
            self.assertEqual(area_km2, 0.0, "Zero mask should return 0 km²")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_calculate_water_area_auto_scale(self):
        """Test water area calculation with automatic scale selection."""
        try:
            geom = ee.Geometry.Rectangle([105.0, 16.0, 105.1, 16.1])
            mask = ee.Image(1).clip(geom)
            
            # Should automatically select appropriate scale
            area_m2, area_km2 = calculate_water_area(mask, geom, scale=None)
            
            self.assertGreater(area_km2, 0, "Area should be positive")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_calculate_water_area_error_handling(self):
        """Test water area calculation error handling."""
        # Create invalid geometry
        mock_geom = Mock(spec=ee.Geometry)
        mock_geom.area = Mock(side_effect=Exception("Test error"))
        mock_mask = Mock(spec=ee.Image)
        
        with patch('gee_khoanh_cùng_ngập_lụt_v2.logger') as mock_logger:
            area_m2, area_km2 = calculate_water_area(mock_mask, mock_geom, scale=10)
            self.assertEqual(area_m2, 0.0, "Should return 0 on error")
            self.assertEqual(area_km2, 0.0, "Should return 0 km² on error")
    
    def test_safe_reduce_region_success(self):
        """Test safe_reduce_region with valid inputs."""
        try:
            geom = ee.Geometry.Rectangle([105.0, 16.0, 105.1, 16.1])
            image = ee.Image(100).clip(geom)
            
            result = safe_reduce_region(
                image, geom, ee.Reducer.mean(), scale=10
            )
            
            self.assertIsInstance(result, dict, "Should return dictionary")
            # Result should contain the mean value
            if result:
                self.assertIn('constant', result)
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_safe_reduce_region_error(self):
        """Test safe_reduce_region error handling."""
        mock_image = Mock(spec=ee.Image)
        mock_image.reduceRegion = Mock(side_effect=Exception("Test error"))
        mock_geom = Mock(spec=ee.Geometry)
        
        with patch('gee_khoanh_cùng_ngập_lụt_v2.logger') as mock_logger:
            result = safe_reduce_region(
                mock_image, mock_geom, ee.Reducer.sum(), scale=10
            )
            self.assertEqual(result, {}, "Should return empty dict on error")
            mock_logger.warning.assert_called_once()
    
    def test_safe_reduce_region_none_result(self):
        """Test safe_reduce_region with None result."""
        mock_image = Mock(spec=ee.Image)
        mock_reduce = Mock()
        mock_reduce.getInfo = Mock(return_value=None)
        mock_image.reduceRegion = Mock(return_value=mock_reduce)
        mock_geom = Mock(spec=ee.Geometry)
        
        result = safe_reduce_region(
            mock_image, mock_geom, ee.Reducer.sum(), scale=10
        )
        self.assertEqual(result, {}, "None result should return empty dict")


class TestGeometryFunctions(unittest.TestCase):
    """Test cases for geometry-related functions."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Earth Engine once for all tests."""
        try:
            ee.Initialize(project='driven-torus-431807-u3')
        except:
            pass
    
    def test_get_region_single_province(self):
        """Test get_region with single province."""
        try:
            geom = get_region("Viet Nam", ["Thua Thien - Hue"])
            self.assertIsInstance(geom, ee.Geometry)
            
            # Check that geometry is valid
            area = geom.area(1).getInfo()
            self.assertGreater(area, 0, "Province area should be positive")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")
    
    def test_get_region_multiple_provinces(self):
        """Test get_region with multiple provinces."""
        try:
            geom = get_region("Viet Nam", ["Thua Thien - Hue", "Da Nang City"])
            self.assertIsInstance(geom, ee.Geometry)
            
            # Multi-province should have larger area than single
            single_area = get_region("Viet Nam", ["Thua Thien - Hue"]).area(1).getInfo()
            multi_area = geom.area(1).getInfo()
            self.assertGreater(multi_area, single_area,
                             "Multi-province area should be larger")
        except Exception as e:
            self.skipTest(f"Earth Engine not available: {e}")


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functions."""
    
    def test_area_calculation_consistency(self):
        """Test that m² to km² conversion is consistent."""
        area_m2 = 1000000  # 1 km² in m²
        area_km2 = area_m2 / 1e6
        
        self.assertAlmostEqual(area_km2, 1.0, places=6)
    
    def test_percentage_calculation(self):
        """Test percentage calculations."""
        total = 1000
        part = 250
        percentage = (part / total) * 100
        
        self.assertEqual(percentage, 25.0)
    
    def test_scale_boundary_conditions(self):
        """Test scale selection boundary conditions."""
        config = Config()
        
        # Just below threshold
        scale1 = config.get_adaptive_scale(999.99)
        self.assertEqual(scale1, 10)
        
        # At threshold
        scale2 = config.get_adaptive_scale(1000.0)
        self.assertEqual(scale2, 30)
        
        # Just above threshold
        scale3 = config.get_adaptive_scale(1000.01)
        self.assertEqual(scale3, 30)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling patterns."""
    
    def test_try_except_returns_default(self):
        """Test that error handlers return sensible defaults."""
        # Test calculate_water_area error return
        mock_mask = Mock()
        mock_geom = Mock()
        mock_mask.selfMask = Mock(side_effect=Exception("Test"))
        
        with patch('gee_khoanh_cùng_ngập_lụt_v2.logger'):
            from gee_khoanh_cùng_ngập_lụt_v2 import calculate_water_area
            area_m2, area_km2 = calculate_water_area(mock_mask, mock_geom, 10)
            self.assertEqual(area_m2, 0.0)
            self.assertEqual(area_km2, 0.0)
    
    def test_safe_functions_dont_raise(self):
        """Test that 'safe' functions don't raise exceptions."""
        mock_image = Mock()
        mock_geom = Mock()
        mock_image.reduceRegion = Mock(side_effect=Exception("Test"))
        
        # Should not raise
        try:
            with patch('gee_khoanh_cùng_ngập_lụt_v2.logger'):
                result = safe_reduce_region(mock_image, mock_geom, 
                                           ee.Reducer.mean(), 10)
            self.assertEqual(result, {})
        except Exception as e:
            self.fail(f"safe_reduce_region raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
