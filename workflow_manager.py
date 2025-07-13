import json
import os
from pathlib import Path
from typing import List, Optional

from core.img_processor import ImageProcessor, OptimizationSettings
from core.gv_api import VisionProcessor
from core.text_analyzer import TextAnalyzer
from data.storage_manager import StorageManager
from data.models import Image, ProcessingStatus


class OCRWorkflowManager:
    """
    Orchestrates the complete OCR workflow while maintaining SRP.
    Coordinates between StorageManager, ImageProcessor, VisionProcessor, and TextAnalyzer.
    """

    def __init__(self, input_folder: str, credentials_path: str, project_root: Optional[str] = None):
        self.input_folder = input_folder
        self.credentials_path = credentials_path
        self.project_root = project_root or str(Path.cwd())

        # Get the input folder name (preserve original structure)
        input_folder_name = Path(input_folder).name

        # Create result folder structure: result/{input_folder_name}/
        self.result_folder = Path(self.project_root) / "result" / input_folder_name

        # Initialize components with proper folder structure
        self.storage_manager = StorageManager(
            input_folder_path=input_folder,
            output_folder=str(self.result_folder / "optimized_images"),
            project_root=project_root
        )
        self.vision_processor = VisionProcessor(credentials_path)
        self.text_analyzer = TextAnalyzer()

        # Create local storage for responses and results inside the specific folder
        self.vision_responses_folder = self.result_folder / "vision_responses"
        self.analysis_results_folder = self.result_folder / "analysis_results"

        # Ensure directories exist
        self.vision_responses_folder.mkdir(parents=True, exist_ok=True)
        self.analysis_results_folder.mkdir(parents=True, exist_ok=True)

    def run_complete_workflow(self, optimization_settings: OptimizationSettings) -> bool:
        """
        Execute the complete OCR workflow:
        1. Discover and create image models with loaded bytes
        2. Optimize images and save to local storage
        3. Submit to Google Cloud Vision API and save responses
        4. Analyze responses and save result strings
        """
        import time

        try:
            print("Starting OCR workflow...")
            start_time = time.time()

            # Step 1: Discover images and create models with loaded bytes
            print("\nStep 1: Discovering images and creating models...")
            step_start = time.time()
            images = self._discover_and_load_images()
            step_duration = time.time() - step_start

            if not images:
                print("No images found in the specified folder.")
                return False
            print(f"Step 1 completed in {step_duration:.1f}s - Found {len(images)} images")

            # Step 2: Optimize images and save to local storage
            print(f"\nStep 2: Optimizing images...")
            step_start = time.time()
            optimized_images = self._optimize_and_save_images(images, optimization_settings)
            step_duration = time.time() - step_start
            print(f"Step 2 completed in {step_duration:.1f}s - Optimized {len(optimized_images)} images")

            # Step 3: Submit to Google Cloud Vision API and save responses
            print(f"\nStep 3: Processing with Google Cloud Vision API...")
            step_start = time.time()
            processed_images = self._process_with_vision_api(optimized_images)
            successful_images = [img for img in processed_images if img.vision_response is not None]
            step_duration = time.time() - step_start
            print(
                f"Step 3 completed in {step_duration:.1f}s - Successfully processed {len(successful_images)} images with Vision API")

            if not successful_images:
                print("No images were successfully processed by Vision API.")
                return False

            # Step 4: Analyze responses and save result strings
            print(f"\nStep 4: Analyzing responses and saving results...")
            step_start = time.time()
            analyzed_images = self._analyze_and_save_results(processed_images)
            step_duration = time.time() - step_start
            completed_analyses = len([img for img in analyzed_images if img.status == ProcessingStatus.COMPLETED])
            print(
                f"Step 4 completed in {step_duration:.1f}s - Analyzed and saved results for {completed_analyses} images")

            total_duration = time.time() - start_time
            print(f"\nOCR workflow completed successfully in {total_duration:.1f}s!")
            return True

        except Exception as e:
            print(f"\nError during OCR workflow: {str(e)}")
            return False
        finally:
            self.storage_manager.close()

    def _discover_and_load_images(self) -> List[Image]:
        """Discover images and create Image objects with loaded bytes."""
        print(f"  → Scanning folder: {self.input_folder}")
        images = self.storage_manager.discover_images()

        if images:
            print(f"  → Found {len(images)} images:")
            total_size = 0
            for i, image in enumerate(images, 1):
                size = len(image.image_bytes) if image.image_bytes else 0
                total_size += size
                print(f"    {i}. {image.filename} ({self._format_size(size)})")

            print(f"  → Total images size: {self._format_size(total_size)}")
        else:
            print(f"  → No supported images found in {self.input_folder}")
            print(f"    Supported formats: JPG, JPEG, PNG")

        return images

    def _optimize_and_save_images(self, images: List[Image], settings: OptimizationSettings) -> List[Image]:
        """Optimize images and save them to local storage, update database."""
        optimized_images = []
        total_images = len(images)

        print(f"  → Starting optimization for {total_images} images...")
        print(
            f"  → Settings: {settings.quality.value}, max_width={settings.max_width}px, quality={settings.jpeg_quality}%")

        for i, image in enumerate(images, 1):
            try:
                print(f"  → Optimizing image {i}/{total_images}: {image.filename}")

                # Update status to processing
                image = self.storage_manager.update_status(image, ProcessingStatus.PROCESSING)

                # Show original size
                original_size = len(image.image_bytes)
                print(f"    Original size: {self._format_size(original_size)}")

                # Optimize image
                optimized_bytes = ImageProcessor.process_image(image.image_bytes, settings)
                optimized_size = len(optimized_bytes)
                compression_ratio = ((original_size - optimized_size) / original_size) * 100

                print(f"    Optimized size: {self._format_size(optimized_size)} ({compression_ratio:.1f}% reduction)")

                # Save optimized image
                optimized_filename = f"opt_{image.filename}"
                image = self.storage_manager.save_image(image, optimized_bytes, optimized_filename)

                optimized_images.append(image)
                print(f"  ✓ Optimized and saved: {image.filename}")

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                print(f"  ✗ Failed to optimize {i}/{total_images} ({image.filename}): {e}")
                continue

        print(f"  → Optimization completed: {len(optimized_images)}/{total_images} images successful")
        return optimized_images

    def _process_with_vision_api(self, images: List[Image]) -> List[Image]:
        """Process optimized images with Google Vision API and save responses."""
        print(f"  → Preparing {len(images)} images for Vision API processing...")

        # Load optimized bytes into Image objects before processing
        successfully_loaded = 0
        for i, image in enumerate(images, 1):
            if image.optimized_file_path and image.status != ProcessingStatus.FAILED:
                try:
                    # Load optimized bytes for Vision API processing
                    optimized_bytes = self.storage_manager.output_storage.read_file_bytes(
                        image.optimized_file_path
                    )
                    image.image_bytes = optimized_bytes
                    successfully_loaded += 1
                    print(f"  → Loaded optimized image {i}/{len(images)}: {image.filename}")
                except Exception as e:
                    print(f"  ✗ Failed to load optimized image {i}/{len(images)} ({image.filename}): {e}")
                    image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)

        print(f"  → Successfully loaded {successfully_loaded} optimized images")
        print(f"  → Sending to Google Vision API...")

        # Use the public interface of VisionProcessor
        processed_images = self.vision_processor.process_images(images)

        print(f"  → Vision API processing completed, saving responses...")

        # Save Vision API responses to local storage
        saved_responses = 0
        for image in processed_images:
            if image.vision_response is not None:
                try:
                    # Save Vision API response as JSON
                    response_filename = f"{Path(image.filename).stem}_vision_response.json"
                    response_path = self.vision_responses_folder / response_filename

                    response_dict = VisionProcessor.get_dict(image.vision_response)
                    with open(response_path, 'w', encoding='utf-8') as f:
                        json.dump(response_dict, f, ensure_ascii=False, indent=2)

                    # Update image model with vision response path
                    image.image_model.vision_json_path = str(response_path.relative_to(Path(self.project_root)))
                    image = self.storage_manager.update_image(image)
                    saved_responses += 1

                except Exception as e:
                    print(f"  ✗ Failed to save Vision API response for {image.filename}: {e}")

        print(f"  → Saved {saved_responses} Vision API responses to storage")
        return processed_images

    def _analyze_and_save_results(self, images: List[Image]) -> List[Image]:
        """Analyze Vision API responses and save individual result strings."""
        analyzed_images = []
        total_images = len(images)
        successful_images = [img for img in images if img.vision_response is not None]

        print(
            f"  → Starting text analysis for {len(successful_images)}/{total_images} images with Vision API responses...")

        for i, image in enumerate(images, 1):
            try:
                if image.vision_response is None:
                    print(f"  → Skipping image {i}/{total_images}: {image.filename} (no Vision API response)")
                    analyzed_images.append(image)
                    continue

                print(f"  → Analyzing image {i}/{total_images}: {image.filename}")

                # Analyze single image (create a list with one image for the analyzer)
                analysis_result = self.text_analyzer.analyze_images([image])

                # Count lines/characters in result
                lines_count = len(analysis_result.split('\n'))
                chars_count = len(analysis_result)
                print(f"    Analysis result: {lines_count} lines, {chars_count} characters")

                # Save analysis result
                result_filename = f"{Path(image.filename).stem}_analysis_result.txt"
                result_path = self.analysis_results_folder / result_filename

                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(analysis_result)

                # Store analysis results in the image object
                image.analysis_results = [analysis_result]

                # Update status to completed
                image = self.storage_manager.update_status(image, ProcessingStatus.COMPLETED)

                analyzed_images.append(image)
                print(f"  ✓ Analyzed and saved: {image.filename}")

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                print(f"  ✗ Analysis failed for {i}/{total_images} ({image.filename}): {e}")
                analyzed_images.append(image)
                continue

        successful_analyses = len([img for img in analyzed_images if img.analysis_results])
        print(f"  → Text analysis completed: {successful_analyses}/{total_images} images successful")
        return analyzed_images

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def get_workflow_summary(self) -> dict:
        """Get summary of workflow processing results."""
        return self.storage_manager.get_processing_summary()