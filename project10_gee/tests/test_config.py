#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Config class.

Tests configuration management, parameter validation,
and adaptive scale calculation logic.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gee_khoanh_cùng_ngập_lụt_v2 import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_config_initialization(self):
        """Test that Config initializes with correct default values."""
        self.assertEqual(self.config.country, "Viet Nam")
        self.assertIsInstance(self.config.provinces, list)
        self.assertEqual(len(self.config.provinces), 2)
        self.assertIn("Thua Thien - Hue", self.config.provinces)
        self.assertIn("Da Nang City", self.config.provinces)
    
    def test_temporal_configuration(self):
        """Test temporal configuration parameters."""
        self.assertEqual(self.config.start_date, "2025-09-01")
        self.assertEqual(self.config.end_date, "2025-11-30")
        
        # Validate date format
        from datetime import datetime
        try:
            datetime.strptime(self.config.start_date, '%Y-%m-%d')
            datetime.strptime(self.config.end_date, '%Y-%m-%d')
        except ValueError:
            self.fail("Date format is incorrect")
    
    def test_flood_detection_thresholds(self):
        """Test flood detection threshold values."""
        self.assertEqual(self.config.ems_threshold, -18.0)
        self.assertLess(self.config.ems_threshold, 0, "EMS threshold should be negative dB")
        self.assertEqual(self.config.adaptive_k, 1.0)
        self.assertGreater(self.config.adaptive_k, 0, "K-factor should be positive")
        self.assertEqual(self.config.baseline_days, 60)
        self.assertEqual(self.config.baseline_vv_threshold, -8.5)
    
    def test_terrain_thresholds(self):
        """Test terrain masking thresholds."""
        self.assertEqual(self.config.hand_threshold, 20)
        self.assertGreater(self.config.hand_threshold, 0, "HAND threshold should be positive")
        self.assertEqual(self.config.slope_threshold, 15)
        self.assertGreater(self.config.slope_threshold, 0, "Slope threshold should be positive")
        self.assertLess(self.config.slope_threshold, 90, "Slope threshold should be < 90 degrees")
    
    def test_post_processing_parameters(self):
        """Test post-processing configuration."""
        self.assertEqual(self.config.kernel_size, 1)
        self.assertEqual(self.config.confidence_thresholds, [80, 60, 40, 20])
        self.assertEqual(self.config.ensemble_vote_threshold, 3)
        self.assertLess(self.config.ensemble_vote_threshold, 6, "Vote threshold should be <= 5 methods")
    
    def test_performance_parameters(self):
        """Test performance optimization parameters."""
        self.assertEqual(self.config.max_pixels, 1e9)
        self.assertEqual(self.config.scale_small_area, 10)
        self.assertEqual(self.config.scale_large_area, 30)
        self.assertLess(self.config.scale_small_area, self.config.scale_large_area,
                       "Small area scale should be finer than large area scale")
    
    def test_adaptive_scale_small_area(self):
        """Test adaptive scale for small areas (<1000 km²)."""
        # Test various small areas
        test_cases = [0.1, 1, 10, 100, 500, 999.9]
        for area in test_cases:
            with self.subTest(area=area):
                scale = self.config.get_adaptive_scale(area)
                self.assertEqual(scale, 10, f"Small area {area} km² should use 10m scale")
    
    def test_adaptive_scale_large_area(self):
        """Test adaptive scale for large areas (≥1000 km²)."""
        # Test various large areas
        test_cases = [1000, 1001, 5000, 10000, 50000]
        for area in test_cases:
            with self.subTest(area=area):
                scale = self.config.get_adaptive_scale(area)
                self.assertEqual(scale, 30, f"Large area {area} km² should use 30m scale")
    
    def test_adaptive_scale_boundary(self):
        """Test adaptive scale at boundary condition (exactly 1000 km²)."""
        scale = self.config.get_adaptive_scale(1000.0)
        self.assertEqual(scale, 30, "1000 km² should use large area scale (30m)")
    
    def test_adaptive_scale_edge_cases(self):
        """Test adaptive scale with edge cases."""
        # Zero area
        scale = self.config.get_adaptive_scale(0)
        self.assertEqual(scale, 10, "Zero area should use small scale")
        
        # Very small area
        scale = self.config.get_adaptive_scale(0.0001)
        self.assertEqual(scale, 10, "Tiny area should use small scale")
        
        # Very large area
        scale = self.config.get_adaptive_scale(1000000)
        self.assertEqual(scale, 30, "Huge area should use large scale")
    
    def test_config_immutability(self):
        """Test that config values can be modified (not immutable)."""
        original_threshold = self.config.ems_threshold
        self.config.ems_threshold = -20.0
        self.assertEqual(self.config.ems_threshold, -20.0)
        # Reset
        self.config.ems_threshold = original_threshold
    
    def test_validation_parameters(self):
        """Test validation and quality parameters."""
        self.assertEqual(self.config.max_cloud_cover, 40)
        self.assertGreaterEqual(self.config.max_cloud_cover, 0)
        self.assertLessEqual(self.config.max_cloud_cover, 100)
        self.assertEqual(self.config.optical_search_days, 7)
    
    def test_max_districts_limit(self):
        """Test maximum districts processing limit."""
        self.assertEqual(self.config.max_districts, 100)
        self.assertGreater(self.config.max_districts, 0, "Max districts should be positive")


class TestConfigScalePerformance(unittest.TestCase):
    """Performance tests for adaptive scale calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_scale_calculation_speed(self):
        """Test that scale calculation is fast (< 1ms)."""
        import time
        
        start = time.time()
        for _ in range(1000):
            self.config.get_adaptive_scale(500.0)
        end = time.time()
        
        avg_time = (end - start) / 1000
        self.assertLess(avg_time, 0.001, "Scale calculation should be < 1ms")


if __name__ == '__main__':
    unittest.main()
