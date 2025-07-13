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

        # Initialize components
        self.storage_manager = StorageManager(
            input_folder_path=input_folder,
            project_root=project_root
        )
        self.vision_processor = VisionProcessor(credentials_path)
        self.text_analyzer = TextAnalyzer()

        # Create local storage for responses and results
        result_folder = Path(self.project_root) / "result" / Path(input_folder).name
        self.vision_responses_folder = result_folder / "vision_responses"
        self.analysis_results_folder = result_folder / "analysis_results"

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
            print("Starting OCR workflow...")

            # Step 1: Discover images and create models with loaded bytes
            print("Step 1: Discovering images and creating models...")
            images = self._discover_and_load_images()
            if not images:
                print("No images found in the specified folder.")
                return False
            print(f"Found {len(images)} images")

            # Step 2: Optimize images and save to local storage
            print("Step 2: Optimizing images...")
            optimized_images = self._optimize_and_save_images(images, optimization_settings)
            print(f"Optimized {len(optimized_images)} images")

            # Step 3: Submit to Google Cloud Vision API and save responses
            print("Step 3: Processing with Google Cloud Vision API...")
            processed_images = self._process_with_vision_api(optimized_images)
            successful_images = [img for img in processed_images if img.vision_response is not None]
            print(f"Successfully processed {len(successful_images)} images with Vision API")

            if not successful_images:
                print("No images were successfully processed by Vision API.")
                return False

            # Step 4: Analyze responses and save result strings
            print("Step 4: Analyzing responses and saving results...")
            analyzed_images = self._analyze_and_save_results(successful_images)
            print(f"Analyzed and saved results for {len(analyzed_images)} images")

            print("OCR workflow completed successfully!")
            return True

        except Exception as e:
            print(f"Error during OCR workflow: {str(e)}")
            return False
        finally:
            self.storage_manager.close()

    def _discover_and_load_images(self) -> List[Image]:
        """Discover images and create Image objects with loaded bytes."""
        return self.storage_manager.discover_images()

    def _optimize_and_save_images(self, images: List[Image], settings: OptimizationSettings) -> List[Image]:
        """Optimize images and save them to local storage, update database."""
        optimized_images = []

        for image in images:
            try:
                # Update status to processing
                image = self.storage_manager.update_status(image, ProcessingStatus.PROCESSING)

                # Optimize image
                optimized_bytes = ImageProcessor.process_image(image.image_bytes, settings)

                # Save optimized image
                optimized_filename = f"opt_{image.filename}"
                image = self.storage_manager.save_image(image, optimized_bytes, optimized_filename)

                optimized_images.append(image)
                print(f"    Optimized: {image.filename}")

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                print(f"    Failed to optimize {image.filename}: {e}")
                continue

        return optimized_images

    def _process_with_vision_api(self, images: List[Image]) -> List[Image]:
        """Process optimized images with Google Vision API and save responses."""
        # Load optimized bytes into Image objects before processing
        for image in images:
            if image.optimized_file_path and image.status != ProcessingStatus.FAILED:
                try:
                    # Load optimized bytes for Vision API processing
                    optimized_bytes = self.storage_manager.output_storage.read_file_bytes(
                        image.optimized_file_path
                    )
                    image.image_bytes = optimized_bytes
                except Exception as e:
                    print(f"    Failed to load optimized image for {image.filename}: {e}")
                    image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)

        # Use the public interface of VisionProcessor
        processed_images = self.vision_processor.process_images(images)

        # Save Vision API responses to local storage
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

                except Exception as e:
                    print(f"    Failed to save Vision API response for {image.filename}: {e}")

        return processed_images

    def _analyze_and_save_results(self, images: List[Image]) -> List[Image]:
        """Analyze Vision API responses and save individual result strings."""
        analyzed_images = []

        for image in images:
            try:
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
                print(f"    Analyzed: {image.filename}")

            except Exception as e:
                # Mark as failed and continue
                image = self.storage_manager.update_status(image, ProcessingStatus.FAILED)
                print(f"    Analysis failed for {image.filename}: {e}")
                analyzed_images.append(image)
                continue

        return analyzed_images

    def get_workflow_summary(self) -> dict:
        """Get summary of workflow processing results."""
        return self.storage_manager.get_processing_summary()