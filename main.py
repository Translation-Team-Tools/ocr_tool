def _generate_output_filename(self) -> str:
    """Generate timestamped filename for output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ocr_results_{timestamp}.txt"  # !/usr/bin/env python3


"""
OCR Tool Main Workflow
Orchestrates image processing, Google Cloud Vision API calls, text analysis, and output generation.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import core modules
from core.image_workflow import ImageWorkflowManager
from core.gv_api import VisionProcessor
from core.text_analyzer import TextAnalyzer
from data.models import Image


class OCRWorkflow:
    """Main OCR workflow orchestrator"""

    def __init__(self, input_folder: str, credentials_path: str):
        self.input_folder = input_folder
        self.credentials_path = credentials_path
        self.result_folder = project_root / "result"
        self.result_folder.mkdir(exist_ok=True)

    def run_workflow(self) -> bool:
        """Execute the complete OCR workflow"""
        try:
            print("Starting OCR processing workflow...")

            # Step 1: Create image objects using ImageWorkflowManager
            print("Step 1: Creating and optimizing image objects...")
            image_objects = ImageWorkflowManager.process_images_workflow(self.input_folder)

            if not image_objects:
                print("No image objects created. Exiting.")
                return False

            print(f"Created and optimized {len(image_objects)} image objects")

            # Step 2: Send to Google Cloud Vision API via VisionProcessor
            print("Step 2: Processing images with Google Cloud Vision API...")
            vision_processor = VisionProcessor(self.credentials_path)
            processed_images = vision_processor.process_images(image_objects)

            # Filter successfully processed images
            successful_images = [img for img in processed_images if img.vision_response is not None]
            print(f"Successfully processed {len(successful_images)} images through Google Vision API")

            if not successful_images:
                print("No images were successfully processed by Vision API. Exiting.")
                return False

            # Step 3: Generate output string via TextAnalyzer
            print("Step 3: Analyzing and generating output string...")
            analyzer = TextAnalyzer()
            output_string = analyzer.analyze_images(successful_images)

            if not output_string:
                print("No output string generated. Exiting.")
                return False

            # Step 4: Write to .txt file in result folder
            print("Step 4: Writing results to file...")
            output_filename = self._generate_output_filename()
            output_path = self.result_folder / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_string)

            print(f"Results written to: {output_path}")
            print("OCR processing workflow completed successfully!")
            return True

        except Exception as e:
            print(f"Error during OCR workflow: {str(e)}")
            return False

    def _create_processing_summary(self, images: List[Image]) -> str:
        """Create a summary of processed images using only public APIs"""
        from output_generation.output_generator import OutputGenerator

        output_generator = OutputGenerator()
        image_sections: List[str] = []

        for image in images:
            try:
                # Create a basic summary section for each image
                summary_lines = [
                    f"✅ OCR Processing completed",
                    f"📁 Original file: {image.original_file_path}",
                    f"🔧 Optimized file: {image.optimized_file_path}",
                    f"📊 Status: {image.status.value}",
                    f"🤖 Vision API response: {'Available' if image.vision_response else 'Not available'}"
                ]

                section = output_generator.build_image_section(
                    lines=summary_lines,
                    filename=image.filename
                )
                image_sections.append(section)

            except Exception as e:
                print(f"    Error creating summary for {image.filename}: {e}")
                continue

        # Generate final output using OutputGenerator's public API
        if image_sections:
            return output_generator.build_final_result(image_sections=image_sections)
        else:
            return "No images were successfully processed."

    """Main OCR workflow orchestrator"""

    def __init__(self, input_folder: str, credentials_path: str):
        self.input_folder = input_folder
        self.credentials_path = credentials_path
        self.result_folder = project_root / "result"
        self.result_folder.mkdir(exist_ok=True)

    def run_workflow(self) -> bool:
        """Execute the complete OCR workflow"""
        try:
            print("Starting OCR processing workflow...")

            # Step 1: Create image objects using ImageWorkflowManager
            print("Step 1: Creating and optimizing image objects...")
            image_objects = ImageWorkflowManager.process_images_workflow(self.input_folder)

            if not image_objects:
                print("No image objects created. Exiting.")
                return False

            print(f"Created and optimized {len(image_objects)} image objects")

            # Step 2: Send to Google Cloud Vision API via VisionProcessor
            print("Step 2: Processing images with Google Cloud Vision API...")
            vision_processor = VisionProcessor(self.credentials_path)
            processed_images = vision_processor.process_images(image_objects)

            # Filter successfully processed images
            successful_images = [img for img in processed_images if img.vision_response is not None]
            print(f"Successfully processed {len(successful_images)} images through Google Vision API")

            if not successful_images:
                print("No images were successfully processed by Vision API. Exiting.")
                return False

            # Step 3: Generate output string via TextAnalyzer (modified to return result)
            print("Step 3: Analyzing and generating output string...")
            output_string = self._analyze_and_generate_output(successful_images)

            if not output_string:
                print("No output string generated. Exiting.")
                return False

            # Step 4: Write to .txt file in result folder
            print("Step 4: Writing results to file...")
            output_filename = self._generate_output_filename()
            output_path = self.result_folder / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_string)

            print(f"Results written to: {output_path}")
            print("OCR processing workflow completed successfully!")
            return True

        except Exception as e:
            print(f"Error during OCR workflow: {str(e)}")
            return False

    def _analyze_and_generate_output(self, images: List[Image]) -> str:
        """
        Modified text analysis that returns the output string.
        Based on TextAnalyzer.analyze_images() but returns the result.
        """
        analyzer = TextAnalyzer()
        image_sections: List[str] = []

        for image in images:
            try:
                # Use the same analysis logic as TextAnalyzer
                paragraphs = analyzer._analyze_full_text_annotation(image.vision_response)
                if paragraphs:
                    section_lines = analyzer._build_output(paragraphs)
                    section = analyzer.output_generator.build_image_section(
                        lines=section_lines,
                        filename=image.filename
                    )
                    image_sections.append(section)
                    print(f"    Analyzed: {image.filename}")
                else:
                    print(f"    No text found in: {image.filename}")

            except Exception as e:
                print(f"    Skipping {image.filename}: Analysis error - {e}")
                continue

        # Generate final output string using OutputGenerator
        if image_sections:
            return analyzer.output_generator.build_final_result(image_sections=image_sections)
        else:
            return ""

    def _generate_output_filename(self) -> str:
        """Generate timestamped filename for output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ocr_results_{timestamp}.txt"


def main():
    """Main entry point"""
    # You'll need to specify these paths
    input_folder = input("Enter input folder path: ").strip()
    if not input_folder:
        print("No input folder specified. Using default...")
        input_folder = str(project_root / "input")  # Default input folder

    # Credentials path - modify this to your actual credentials file
    credentials_path = str(project_root / "credentials.json")

    # Check if credentials exist
    if not os.path.exists(credentials_path):
        print(f"Warning: Google credentials not found at {credentials_path}")
        print("Please ensure your Google Vision API credentials are properly configured.")
        credentials_path = input("Enter path to Google credentials JSON file (or press Enter to continue): ").strip()
        if not credentials_path:
            credentials_path = str(project_root / "config" / "google_credentials.json")

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        return False

    # Run workflow
    workflow = OCRWorkflow(input_folder, credentials_path)
    success = workflow.run_workflow()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)