#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Script for Keyword-based Disaster Extraction
Script tiện ích để chạy các demo
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Keyword-based Disaster Extraction Runner')
    parser.add_argument('--demo', choices=['simple', 'full'],
                       default='full', help='Chọn demo để chạy')
    parser.add_argument('--output', type=str,
                       help='Thư mục output (mặc định: ../data)')

    args = parser.parse_args()

    # Set output directory
    if args.output:
        os.environ['KEYWORD_OUTPUT_DIR'] = args.output

    # Import and run demo
    if args.demo == 'simple':
        from scripts.demo_simple import demo as run_demo_func
    else:
        from scripts.demo_full import run_demo as run_demo_func

    # Execute demo
    run_demo_func()

if __name__ == "__main__":
    main()