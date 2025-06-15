import json
from pathlib import Path

from database import DatabaseManager
from gv_api import VisionAPIClient
from image_processor import ImageProcessor
from output_generator import OutputGenerator
from progress_tracker import ProgressTracker
from text_analyzer import TextAnalyzer
from error_handler import ErrorHandler


class OCRBatchProcessor:
    """Main application class that orchestrates the OCR processing workflow."""

    def __init__(self, credentials_path: str = None):
        # Resolve default credentials path if not provided
        if credentials_path is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            credentials_path = os.path.join(script_dir, "credentials.json")
            print(f"Using default credentials: {credentials_path}")

        self.db_manager = DatabaseManager()
        self.image_processor = ImageProcessor(self.db_manager)
        self.vision_client = VisionAPIClient(credentials_path)
        self.text_analyzer = TextAnalyzer()
        self.output_generator = OutputGenerator()
        self.error_handler = ErrorHandler()

    def process_folder(self, input_folder: str) -> bool:
        """Main processing workflow."""
        input_path = Path(input_folder)
        result_path = Path("result")

        if not input_path.exists():
            print(f"Error: Input folder '{input_folder}' does not exist.")
            return False

        # Stage 1: Discover files
        print("=== Stage 1/5: Discovering Files ===")
        image_files = self.image_processor.discover_images(input_folder)

        if not image_files:
            print("No supported image files found.")
            return False

        print(f"Found {len(image_files)} images")
        progress = ProgressTracker(len(image_files))

        # Stage 2: Copy images and check for duplicates
        progress.update_stage(2)
        images_to_process = []

        for i, image_path in enumerate(image_files, 1):
            progress.update_image(i, str(image_path))

            try:
                file_hash = self.image_processor.calculate_file_hash(image_path)
                relative_path = str(image_path.relative_to(input_path))

                if self.db_manager.is_processed(relative_path, file_hash):
                    print(f"  Skipping (already processed): {image_path.name}")
                    continue

                copied_path = self.image_processor.copy_image(image_path, input_path, result_path)
                images_to_process.append({
                    'original_path': image_path,
                    'copied_path': copied_path,
                    'relative_path': relative_path,
                    'file_hash': file_hash
                })

            except Exception as e:
                error_type = self.error_handler.categorize_error(e)
                self.error_handler.add_error(str(image_path), error_type, str(e))
                print(f"  Error copying {image_path.name}: {e}")

        if not images_to_process:
            print("All images already processed.")
            return True

        # Stage 3: OCR Processing
        progress.update_stage(3)
        processed_results = []

        for i, image_info in enumerate(images_to_process, 1):
            progress.update_image(i, image_info['relative_path'])

            try:
                # Process with Vision API
                vision_response = self.vision_client.process_image(image_info['copied_path'])

                # Save JSON response
                json_path = result_path / "gv_results" / f"{image_info['relative_path']}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(vision_response, f, ensure_ascii=False, indent=2)

                # Mark as processed in database
                self.db_manager.add_processed_image(
                    image_info['relative_path'],
                    image_info['original_path'].name,
                    image_info['file_hash'],
                    str(json_path),
                    'completed'
                )

                processed_results.append({
                    'relative_path': image_info['relative_path'],
                    'vision_response': vision_response
                })

            except Exception as e:
                error_type = self.error_handler.categorize_error(e)
                self.error_handler.add_error(image_info['relative_path'], error_type, str(e))
                self.db_manager.add_processed_image(
                    image_info['relative_path'],
                    image_info['original_path'].name,
                    image_info['file_hash'],
                    '',
                    'failed',
                    str(e)
                )
                print(f"  Error processing {image_info['original_path'].name}: {e}")

        # Stage 4: Analyze results
        progress.update_stage(4)
        final_results = []

        for result in processed_results:
            analyzed = self.text_analyzer.analyze_vision_response(result['vision_response'])
            final_results.append({
                'relative_path': result['relative_path'],
                'regular_text': analyzed['regular_paragraphs'],
                'furigana_text': analyzed['furigana_paragraphs']
            })

        # Stage 5: Generate output
        progress.update_stage(5)
        self.output_generator.generate_text_output(final_results)

        # Handle failed images
        retry_images = self.error_handler.prompt_retry_failed()
        if retry_images:
            print("\nRetrying failed images...")
            # Here you could implement retry logic

        print(f"\nProcessing complete! Processed {len(final_results)} images successfully.")
        if self.error_handler.failed_images:
            print(f"Failed: {len(self.error_handler.failed_images)} images")

        return True