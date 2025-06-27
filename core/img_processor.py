import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from data.models import Image, ProcessingStatus
from data.local_storage import LocalStorage  # Import the LocalStorage class


class OCRQuality(Enum):
    """Simple quality options for OCR optimization."""
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


@dataclass
class OptimizationSettings:
    """OCR optimization settings with user-friendly options."""
    quality: OCRQuality = OCRQuality.BALANCED
    enhance_contrast: bool = True
    convert_to_grayscale: bool = False
    max_width: int = 2048
    jpeg_quality: int = 90

    def __post_init__(self):
        """Adjust settings based on quality preset."""
        if self.quality == OCRQuality.FAST:
            self.max_width = 1600
            self.jpeg_quality = 85
        elif self.quality == OCRQuality.BEST:
            self.max_width = 3000
            self.jpeg_quality = 95


class ImageProcessor:
    """
    Processes images for optimal OCR recognition with Google Vision API.
    Now uses LocalStorage for all file system operations.
    """

    def __init__(self, output_folder: str = "optimized_images"):
        """
        Initialize the image processor.

        Args:
            output_folder: Where to save optimized images
        """
        self.output_folder = Path(output_folder)
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.settings = OptimizationSettings()

        # Initialize LocalStorage with supported formats
        self.storage = LocalStorage()

    def configure_optimization(self,
                               quality: OCRQuality = OCRQuality.BALANCED,
                               enhance_contrast: bool = True,
                               convert_to_grayscale: bool = False) -> None:
        """
        Configure how images should be optimized for OCR.

        Args:
            quality: Overall quality preset (FAST/BALANCED/BEST)
            enhance_contrast: Make text more readable (recommended: True)
            convert_to_grayscale: Convert to grayscale (good for documents)
        """
        self.settings = OptimizationSettings(
            quality=quality,
            enhance_contrast=enhance_contrast,
            convert_to_grayscale=convert_to_grayscale
        )

    def discover_and_create_images(self, input_folder_path: str) -> List[Image]:
        """
        Find all images in folder and create Image objects.

        Args:
            input_folder_path: Path to folder containing images

        Returns:
            List of Image objects with metadata
        """
        # Use LocalStorage to validate path
        input_path = self.storage.validate_path_exists(input_folder_path)

        # Use LocalStorage to discover images
        image_files = self.storage.discover_files_recursive(input_path)
        images_list = []

        print(f"Found {len(image_files)} images in {input_folder_path}")

        for file_path in image_files:
            try:
                # Use LocalStorage to calculate relative path
                relative_path = self.storage.get_relative_path(file_path, input_path)

                # Create Image object
                image = Image(
                    filename=file_path.name,
                    original_file_path=str(relative_path),
                    file_hash=self.storage.calculate_file_hash(file_path),
                    status=ProcessingStatus.PENDING
                )

                images_list.append(image)

            except Exception as e:
                print(f"Warning: Could not process {file_path.name}: {e}")

        return images_list

    def optimize_images(self, images_list: List[Image], input_folder_path: str) -> List[Image]:
        """
        Optimize images for OCR and save to output folder.

        Args:
            images_list: List of Image objects to process
            input_folder_path: Original folder containing the images

        Returns:
            List of Image objects with updated status and paths
        """
        input_path = Path(input_folder_path)
        processed_images_list = []

        print(f"Optimizing {len(images_list)} images with {self.settings.quality.value} quality...")
        print(f"Settings: contrast={self.settings.enhance_contrast}, "
              f"grayscale={self.settings.convert_to_grayscale}")

        for i, image in enumerate(images_list, 1):
            print(f"  [{i}/{len(images_list)}] Processing {image.filename}...")

            try:
                # Construct full source path
                source_path = input_path / image.original_file_path

                # Use LocalStorage to optimize and save image
                optimized_path = self._optimize_and_save_image(source_path, image.original_file_path)

                # Update image model
                image.optimized_file_path = str(self.storage.get_relative_path(optimized_path, self.output_folder))
                image.status = ProcessingStatus.COMPLETED

                processed_images_list.append(image)

            except Exception as e:
                # Handle processing errors
                image.status = ProcessingStatus.FAILED
                image.error_message = str(e)
                processed_images_list.append(image)
                print(f"    Error: {e}")

        successful = len([img_obj for img_obj in processed_images_list if img_obj.status == ProcessingStatus.COMPLETED])
        print(f"\nCompleted: {successful}/{len(images_list)} images optimized successfully")

        return processed_images_list

    def _optimize_and_save_image(self, source_path: Path, relative_path: str) -> Path:
        """Optimize single image for OCR and save using LocalStorage."""
        # Use LocalStorage to create output path (force .jpg extension)
        destination = self.storage.create_output_path(
            str(self.output_folder),
            relative_path,
            force_extension='.jpg'
        )

        # Use LocalStorage to check if file already exists
        if self.storage.file_exists(destination):
            print(f"    Skipped: Already exists")
            return destination

        # Use LocalStorage to optimize and save
        self._ocr_optimize_and_save(source_path, destination)
        return destination

    def _ocr_optimize_and_save(self, source_path: Path, destination: Path):
        """Optimize image specifically for OCR processing using LocalStorage."""
        try:
            # Get original image dimensions for reporting
            with self.storage.open_image(source_path) as img_obj:
                original_dimensions = img_obj.size

            # Use LocalStorage to optimize image
            original_size, optimized_size, processing_time = self.storage.optimize_image_for_ocr(
                source_path=source_path,
                destination=destination,
                max_width=self.settings.max_width,
                jpeg_quality=self.settings.jpeg_quality,
                enhance_contrast=self.settings.enhance_contrast,
                convert_to_grayscale=self.settings.convert_to_grayscale
            )

            # Get new dimensions for reporting
            with self.storage.open_image(destination) as img_obj:
                new_dimensions = img_obj.size

            # Prepare enhancement list for reporting
            enhancements = []
            if self.settings.enhance_contrast:
                enhancements.append("contrast+")
            if self.settings.convert_to_grayscale:
                enhancements.append("grayscale")

            # Use LocalStorage to print optimization stats
            self.storage.print_optimization_stats(
                original_size=original_size,
                optimized_size=optimized_size,
                processing_time=processing_time,
                original_dimensions=original_dimensions,
                new_dimensions=new_dimensions,
                enhancements=enhancements
            )

        except Exception as e:
            print(f"    Warning: Could not optimize {source_path.name}, copying original: {e}")
            # Use LocalStorage for fallback copy
            self.storage.copy_file(source_path, destination)

            # Print simple copy stats
            original_size = self.storage.get_file_size(source_path)
            size_mb = self.storage.format_file_size(original_size)
            print(f"    Copied: {size_mb} (no optimization)")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = ImageProcessor("optimized_for_ocr")

    # Configure for best OCR results
    processor.configure_optimization(
        quality=OCRQuality.BEST,
        enhance_contrast=True,
        convert_to_grayscale=False  # Set True for document scans
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to the folder')
    args = parser.parse_args()

    # Process images
    input_folder = args.input_path

    # Discover and create Image objects
    images = processor.discover_and_create_images(input_folder)

    # Optimize for OCR
    processed_images = processor.optimize_images(images, input_folder)

    # Check results
    for img in processed_images:
        if img.status == ProcessingStatus.COMPLETED:
            print(f"✓ {img.filename} -> {img.optimized_file_path}")
        else:
            print(f"✗ {img.filename}: {img.error_message}")