import json
import os
import time
from pathlib import Path
from typing import List, Optional

from core.img_processor import ImageProcessor, OptimizationSettings
from core.gv_api import VisionProcessor
from core.text_analyzer import TextAnalyzer
from data.storage_manager import StorageManager
from data.models import Image, ProcessingStatus
from utils.logger import logger


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
        try:
            logger.step_header(1, "Discovering Images", 4)
            images = self._discover_and_load_images()
            if not images:
                logger.error("No images found in the specified folder")
                return False

            logger.step_header(2, "Optimizing Images", 4)
            optimized_images = self._optimize_and_save_images(images, optimization_settings)

            logger.step_header(3, "Processing with Vision API", 4)
            processed_images = self._process_with_vision_api(optimized_images)
            successful_images = [img for img in processed_images if img.vision_response is not None]

            if not successful_images:
                logger.error("No images were successfully processed by Vision API")
                return False

            logger.step_header(4, "Analyzing Results", 4)
            analyzed_images = self._analyze_and_save_results(processed_images)

            logger.success("OCR workflow completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error during OCR workflow: {str(e)}")
            return False
        finally:
            self.storage_manager.close()

    def _discover_and_load_images(self) -> List[Image]:
        """Discover images and create Image objects with loaded bytes."""
        logger.status(f"Scanning {Path(self.input_folder).name}")
        images = self.storage_manager.discover_images()

        if images:
            total_size = sum(len(img.image_bytes) if img.image_bytes else 0 for img in images)
            logger.success(f"Found {len(images)} images ({logger.format_size(total_size)} total)")
        else:
            logger.warning(f"No supported images found (JPG, JPEG, PNG)")

        return images

    def _optimize_and_save_images(self, images: List[Image], settings: OptimizationSettings) -> List[Image]:
        """Optimize images and save them to local storage, update database."""
        optimized_images = []
        total_images = len(images)
        total_original_size = 0
        total_optimized_size = 0

        logger.info(f"Optimizing {total_images} images with {settings.quality.value} quality")

        for i, image in enumerate(images, 1):
            try:
                logger.progress(i, total_images, image.filename)

                # Update status to processing
                image = self.storage_manager.update_status(image, ProcessingStatus.PROCESSING)

                # Track sizes
                original_size = len(image.image_bytes)
                total_original_size += original_size

                # Optimize image
                optimized_bytes = ImageProcessor.process_image(image.image_bytes, settings)
                optimized_size = len(optimized_bytes)
                total_optimized_size += optimized_size

                # Save optimized image
                optimized_filename = f"opt_{image.filename}"
                image = self.storage_manager.save_image(image, optimized_bytes, optimized_filename)
                optimized_images.append(image)

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                logger.error(f"Failed to optimize {image.filename}: {e}")
                continue

        # Complete progress and show overall compression results
        if total_original_size > 0:
            reduction = ((total_original_size - total_optimized_size) / total_original_size) * 100
            logger.progress_complete(
                f"Optimization complete: {len(optimized_images)}/{total_images} images optimized ({reduction:.1f}% size reduction)")
        else:
            logger.progress_complete(f"Optimization complete: {len(optimized_images)}/{total_images} images")

        return optimized_images

    def _process_with_vision_api(self, images: List[Image]) -> List[Image]:
        """Process optimized images with Google Vision API and save responses."""
        # Load optimized bytes into Image objects before processing
        successfully_loaded = 0
        for image in images:
            if image.optimized_file_path and image.status != ProcessingStatus.FAILED:
                try:
                    # Load optimized bytes for Vision API processing
                    optimized_bytes = self.storage_manager.output_storage.read_file_bytes(
                        image.optimized_file_path
                    )
                    image.image_bytes = optimized_bytes
                    successfully_loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load optimized image {image.filename}: {e}")
                    image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)

        logger.info(f"Loaded {successfully_loaded} optimized images for Vision API")

        # Use the public interface of VisionProcessor but handle progress in workflow
        logger.info(f"Processing with Vision API...")

        processed_images = []
        total_to_process = len([img for img in images if img.image_bytes and img.status != ProcessingStatus.FAILED])

        if total_to_process == 0:
            return images

        current_processed = 0

        for image in images:
            if not image.image_bytes or image.status == ProcessingStatus.FAILED:
                processed_images.append(image)
                continue

            current_processed += 1
            logger.progress(current_processed, total_to_process, image.filename)

            try:
                # Process single image list with Vision API
                single_result = self.vision_processor.process_images([image])
                processed_images.append(single_result[0])
            except Exception as e:
                image.vision_response = None
                processed_images.append(image)
                print(f"   OCR failed for {image.filename}: {e}")

        successful_vision = len([img for img in processed_images if img.vision_response is not None])
        logger.progress_complete(f"Vision API complete: {successful_vision}/{total_to_process} images processed")

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
                    logger.error(f"Failed to save Vision API response for {image.filename}: {e}")

        if saved_responses > 0:
            logger.success(f"Saved {saved_responses} Vision API responses")

        return processed_images

    def _analyze_and_save_results(self, images: List[Image]) -> List[Image]:
        """Analyze Vision API responses and save individual result strings."""
        analyzed_images = []
        total_images = len(images)
        successful_images = [img for img in images if img.vision_response is not None]

        logger.info(f"Analyzing text from {len(successful_images)} processed images")

        for i, image in enumerate(images, 1):
            try:
                if image.vision_response is None:
                    analyzed_images.append(image)
                    continue

                logger.progress(i, total_images, image.filename)

                # Analyze single image (create a list with one image for the analyzer)
                analysis_result = self.text_analyzer.analyze_images([image])

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

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                logger.error(f"Analysis failed for {image.filename}: {e}")
                analyzed_images.append(image)
                continue

        successful_analyses = len([img for img in analyzed_images if img.analysis_results])
        logger.progress_complete(f"Text analysis complete: {successful_analyses}/{total_images} results saved")
        return analyzed_images

    def get_workflow_summary(self) -> dict:
        """Get summary of workflow processing results."""
        return self.storage_manager.get_processing_summary()