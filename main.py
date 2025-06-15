#!/usr/bin/env python3
"""
Japanese OCR Batch Processor
A CLI tool for batch OCR processing of Japanese text images using Google Vision API
with furigana detection, confidence marking, and progress tracking.
"""

import argparse
import sys

from batch_processor import OCRBatchProcessor

try:
    from google.cloud import vision
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install google-cloud-vision Pillow tqdm")
    sys.exit(1)

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Japanese OCR Batch Processor using Google Vision API"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing images to process"
    )
    parser.add_argument(
        "--credentials",
        help="Path to Google Cloud service account JSON file"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="1.0.0"
    )

    args = parser.parse_args()

    print("Japanese OCR Batch Processor v1.0.0")
    print("=" * 50)

    processor = OCRBatchProcessor(args.credentials)
    success = processor.process_folder(args.input_folder)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()