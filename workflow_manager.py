from pathlib import Path
from core.batch_processor import BatchProcessor
from core.gv_api import VisionProcessor
from text_analyzer import TextAnalyzer
from output_generator import OutputGenerator
from database import DatabaseManager
from models.models import ProcessingStatus


class WorkflowManager:
    """Orchestrates the complete OCR processing workflow."""

    def __init__(self, input_folder: str, credentials_path: str):
        self.input_folder = input_folder
        self.credentials_path = credentials_path

        # Initialize modules
        self.batch_processor = BatchProcessor()
        self.vision_processor = VisionProcessor(credentials_path)
        self.text_analyzer = TextAnalyzer()
        self.output_generator = OutputGenerator()
        self.db_manager = DatabaseManager()

        # Statistics tracking
        self.stats = {
            'total_discovered': 0,
            'total_processed': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_skipped': 0
        }

    def process_folder(self):
        """Execute the complete OCR processing workflow."""
        print("Japanese OCR Batch Processor")
        print("=" * 50)

        try:
            # Validate input
            if not self._validate_input():
                return

            # Stage 1: Image Discovery and Optimization
            print("\n=== Stage 1/4: Discovering and Optimizing Images ===")
            images = self._stage_1_discovery_and_optimization()
            if not images:
                print("No images to process. Exiting.")
                return

            # Stage 2: OCR Processing
            print("\n=== Stage 2/4: Processing Images with Google Vision API ===")
            ocr_images = self._stage_2_ocr_processing(images)

            # Stage 3: Text Analysis
            print("\n=== Stage 3/4: Analyzing Text and Detecting Furigana ===")
            analyzed_results = self._stage_3_text_analysis(ocr_images)

            # Stage 4: Output Generation
            print("\n=== Stage 4/4: Generating Output File ===")
            self._stage_4_output_generation(analyzed_results)

            # Final reporting
            self._final_report()

        except Exception as e:
            print(f"\nCritical error in workflow: {e}")
        finally:
            # Always close database connection
            print("\nClosing database connection...")
            self.db_manager.close()

    def _validate_input(self) -> bool:
        """Validate input parameters."""
        input_path = Path(self.input_folder)
        if not input_path.exists():
            print(f"Error: Input folder '{self.input_folder}' does not exist.")
            return False

        credentials_path = Path(self.credentials_path)
        if not credentials_path.exists():
            print(f"Error: Credentials file '{self.credentials_path}' does not exist.")
            return False

        return True

    def _stage_1_discovery_and_optimization(self):
        """Stage 1: Discover images and optimize them."""
        try:
            images = self.batch_processor.process_folder(self.input_folder)

            self.stats['total_discovered'] = len(images)
            completed_count = len([img for img in images if img.status == ProcessingStatus.COMPLETED])
            pending_count = len([img for img in images if img.status == ProcessingStatus.PENDING])
            failed_count = len([img for img in images if img.status == ProcessingStatus.FAILED])

            print(f"Discovery complete:")
            print(f"  Total images found: {self.stats['total_discovered']}")
            print(f"  Already processed: {completed_count}")
            print(f"  New/changed images: {pending_count}")
            print(f"  Failed optimization: {failed_count}")

            if failed_count > 0:
                print("  Optimization errors occurred for some images.")

            return images

        except Exception as e:
            print(f"Error in discovery and optimization stage: {e}")
            return []

    def _stage_2_ocr_processing(self, images):
        """Stage 2: Process images with Google Vision API."""
        try:
            # Filter images that need OCR processing
            images_for_ocr = [img for img in images if img.status == ProcessingStatus.PENDING]

            if not images_for_ocr:
                print("No new images to process with OCR.")
                return images

            print(f"Processing {len(images_for_ocr)} images with Google Vision API...")

            # Process with Vision API
            ocr_results = self.vision_processor.process_images(images_for_ocr)

            # Update database with OCR results
            print("Updating database with OCR results...")
            for image in ocr_results:
                try:
                    self.db_manager.save_image(image)
                except Exception as e:
                    print(f"  Database error for {image.filename}: {e}")

            # Combine with already processed images
            all_images = []
            processed_paths = {img.file_path for img in ocr_results}

            # Add newly processed images
            all_images.extend(ocr_results)

            # Add already completed images that weren't reprocessed
            for img in images:
                if img.file_path not in processed_paths and img.status == ProcessingStatus.COMPLETED:
                    all_images.append(img)

            completed_count = len([img for img in all_images if img.status == ProcessingStatus.COMPLETED])
            failed_count = len([img for img in all_images if img.status == ProcessingStatus.FAILED])

            print(f"OCR processing complete:")
            print(f"  Successfully processed: {completed_count}")
            print(f"  Failed: {failed_count}")

            return all_images

        except Exception as e:
            print(f"Error in OCR processing stage: {e}")
            return images

    def _stage_3_text_analysis(self, images):
        """Stage 3: Analyze text and detect furigana."""
        try:
            # Filter successfully processed images
            completed_images = [img for img in images if img.status == ProcessingStatus.COMPLETED]

            if not completed_images:
                print("No successfully processed images to analyze.")
                return []

            print(f"Analyzing text from {len(completed_images)} images...")

            analyzed_results = self.text_analyzer.analyze_images(completed_images)

            print(f"Text analysis complete:")
            print(f"  Images analyzed: {len(analyzed_results)}")
            print(f"  Images skipped: {len(completed_images) - len(analyzed_results)}")

            return analyzed_results

        except Exception as e:
            print(f"Error in text analysis stage: {e}")
            return []

    def _stage_4_output_generation(self, analyzed_results):
        """Stage 4: Generate final output file."""
        try:
            if not analyzed_results:
                print("No analyzed results to generate output.")
                return

            print(f"Generating output file from {len(analyzed_results)} analyzed images...")

            self.output_generator.generate_output(analyzed_results)

            print("Output generation complete.")

        except Exception as e:
            print(f"Error in output generation stage: {e}")

    def _final_report(self):
        """Generate final processing report."""
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)

        # Get final statistics from database
        try:
            all_images = self.db_manager.get_all_images()
            completed_images = [img for img in all_images if img.status == ProcessingStatus.COMPLETED]
            failed_images = [img for img in all_images if img.status == ProcessingStatus.FAILED]

            print(f"Total images discovered: {len(all_images)}")
            print(f"Successfully processed: {len(completed_images)}")
            print(f"Failed: {len(failed_images)}")

            if failed_images:
                print("\nFailed images:")
                for img in failed_images:
                    error_msg = img.error_message or "Unknown error"
                    print(f"  - {img.filename}: {error_msg}")

        except Exception as e:
            print(f"Error generating final report: {e}")