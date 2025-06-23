#!/usr/bin/env python3
"""
Japanese OCR Image Processor
Processes images containing Japanese text using Google Cloud Vision API
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
from google.cloud import vision
from google.oauth2 import service_account


class JapaneseOCRProcessor:
    def __init__(self, credentials_path):
        """Initialize the OCR processor with Google Cloud credentials"""
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = vision.ImageAnnotatorClient(credentials=credentials)

    def optimize_image(self, image_path, max_size_mb=4, max_dimension=4096):
        """
        Optimize image size for API processing

        Args:
            image_path: Path to the image file
            max_size_mb: Maximum file size in MB
            max_dimension: Maximum width or height in pixels

        Returns:
            Optimized image as bytes
        """
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

            # Resize if dimensions are too large
            if img.width > max_dimension or img.height > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # Save to bytes with optimization
            output = io.BytesIO()
            quality = 95

            while True:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size_mb = output.tell() / (1024 * 1024)

                if size_mb <= max_size_mb or quality <= 30:
                    break

                quality -= 5

            output.seek(0)
            return output.read()

    def process_image(self, image_path):
        """
        Process a single image and extract Japanese text

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing text results and confidence data
        """
        try:
            # Optimize image
            optimized_image = self.optimize_image(image_path)

            # Create vision image object
            image = vision.Image(content=optimized_image)

            # Perform text detection with Japanese language hint
            response = self.client.document_text_detection(
                image=image,
                image_context={'language_hints': ['ja']}
            )

            if response.error.message:
                return {
                    'error': response.error.message,
                    'text': '',
                    'low_confidence_chars': []
                }

            # Extract text and analyze confidence
            full_text = response.full_text_annotation
            text_content = full_text.text if full_text else ''

            low_confidence_chars = []

            # Analyze character-level confidence
            if full_text and full_text.pages:
                for page in full_text.pages:
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                for symbol in word.symbols:
                                    confidence = symbol.confidence
                                    if confidence < 0.8:  # Mark as low confidence if below 80%
                                        char = symbol.text
                                        low_confidence_chars.append({
                                            'char': char,
                                            'confidence': confidence,
                                            'position': len(low_confidence_chars)
                                        })

            return {
                'text': text_content,
                'low_confidence_chars': low_confidence_chars,
                'error': None
            }

        except Exception as e:
            return {
                'error': str(e),
                'text': '',
                'low_confidence_chars': []
            }

    def process_folder(self, folder_path, output_path):
        """
        Process all images in a folder

        Args:
            folder_path: Path to the folder containing images
            output_path: Path for the output text file
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

        # Get all image files
        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        # Sort files for consistent ordering
        image_files.sort()

        print(f"Found {len(image_files)} images to process")

        # Process each image and collect results
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            result = self.process_image(image_file)
            results.append({
                'filename': image_file.name,
                'result': result
            })

        # Write results to output file
        self.write_results(results, output_path)
        print(f"\nResults saved to: {output_path}")

    def write_results(self, results, output_path):
        """
        Write OCR results to a formatted text file

        Args:
            results: List of result dictionaries
            output_path: Path for the output file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("Japanese OCR Results\n")
            f.write(f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            # Write results for each image
            for i, item in enumerate(results, 1):
                filename = item['filename']
                result = item['result']

                f.write(f"[Image {i}] {filename}\n")
                f.write("-" * 60 + "\n")

                if result['error']:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    # Write extracted text
                    if result['text']:
                        f.write("Extracted Text:\n")
                        f.write(result['text'])
                        f.write("\n\n")
                    else:
                        f.write("No text detected in this image.\n\n")

                    # Write low confidence characters if any
                    if result['low_confidence_chars']:
                        f.write("Low Confidence Characters:\n")
                        for lc in result['low_confidence_chars']:
                            f.write(f"  - '{lc['char']}' (confidence: {lc['confidence']:.2%})\n")
                        f.write("\n")

                f.write("=" * 80 + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Process Japanese text from images using Google Cloud Vision API'
    )
    parser.add_argument(
        'folder_path',
        help='Path to folder containing images'
    )
    parser.add_argument(
        'credentials_path',
        help='Path to Google Cloud service account credentials JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        default='ocr_results.txt',
        help='Output file path (default: ocr_results.txt)'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder not found: {args.folder_path}")
        return

    if not os.path.isfile(args.credentials_path):
        print(f"Error: Credentials file not found: {args.credentials_path}")
        return

    # Create processor and process folder
    try:
        processor = JapaneseOCRProcessor(args.credentials_path)
        processor.process_folder(args.folder_path, args.output)
    except Exception as e:
        print(f"Error initializing OCR processor: {e}")
        return


if __name__ == "__main__":
    main()