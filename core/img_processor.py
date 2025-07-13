import argparse
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from PIL import Image as PILImage

from data.models import Image, ProcessingStatus
from data.storage_manager import StorageManager


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
    Now uses StorageManager for unified storage operations.
    """

    def __init__(self, input_folder_path: str, output_folder: str = "optimized_images", project_root: str = None):
        """
        Initialize the image processor.

        Args:
            input_folder_path: Path to folder containing images
            output_folder: Where to save optimized images
            project_root: Project root for database location
        """
        self.settings = OptimizationSettings()

        # Initialize StorageManager (handles both file system and database)
        self.storage_manager = StorageManager(
            input_folder_path=input_folder_path,
            output_folder=output_folder,
            project_root=project_root
        )

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

    def discover_and_save_images(self) -> List[Image]:
        """
        Discover all images and save them to database.

        Returns:
            List of Image objects saved to database
        """
        return self.storage_manager.discover_and_save_images()

    def optimize_image(self, image: Image, max_width: int = 2048, jpeg_quality: int = 90,
                       enhance_contrast: bool = True, convert_to_grayscale: bool = False) -> Image:
        """
        Optimize a single image and update database record.
        """
        try:
            # Get fresh copy from database to avoid session issues
            fresh_image = self.storage_manager.database.get_image_by_id(image.id)
            if not fresh_image:
                raise ValueError(f"Image with ID {image.id} not found in database")

            # Update status to processing
            fresh_image.status = ProcessingStatus.PROCESSING
            fresh_image = self.storage_manager.database.update_image(fresh_image)

            # Create optimized output path (change extension to .jpg)
            source_path = fresh_image.original_file_path
            optimized_relative_path = self._create_optimized_path(fresh_image.original_file_path)

            # Check if already optimized
            if self.storage_manager.output_storage.file_exists(optimized_relative_path):
                print(f"    Skipped: {fresh_image.filename} already optimized")
                # Update database with existing path
                fresh_image.optimized_file_path = optimized_relative_path
                fresh_image.status = ProcessingStatus.COMPLETED
                fresh_image.processed_at = datetime.now()
                return self.storage_manager.database.update_image(fresh_image)

            # Get original file dimensions and size before optimization
            original_size = self.storage_manager.input_storage.get_file_size(source_path)

            # Read original image to get dimensions
            with self.storage_manager.input_storage.fs.open(source_path, 'rb') as f:
                orig_img = PILImage.open(BytesIO(f.read()))
                original_dimensions = orig_img.size

            # First copy file to output storage, then optimize it there
            self.storage_manager.output_storage.ensure_directory_exists(optimized_relative_path)

            # Copy original to output storage temporarily
            temp_path = optimized_relative_path + ".temp"
            with self.storage_manager.input_storage.fs.open(source_path, 'rb') as src:
                with self.storage_manager.output_storage.fs.open(temp_path, 'wb') as dst:
                    dst.write(src.read())

            # Optimize image using output storage
            original_size, optimized_size, processing_time = self.storage_manager.output_storage.optimize_image_for_ocr(
                source_path=temp_path,
                destination=optimized_relative_path,
                max_width=max_width,
                jpeg_quality=jpeg_quality,
                enhance_contrast=enhance_contrast,
                convert_to_grayscale=convert_to_grayscale
            )

            # Clean up temp file
            if self.storage_manager.output_storage.file_exists(temp_path):
                self.storage_manager.output_storage.delete_file(temp_path)

            # Update Image object with results
            fresh_image.optimized_file_path = optimized_relative_path
            fresh_image.status = ProcessingStatus.COMPLETED
            fresh_image.processed_at = datetime.now()

            # Get new dimensions
            with self.storage_manager.output_storage.fs.open(optimized_relative_path, 'rb') as f:
                opt_img = PILImage.open(BytesIO(f.read()))
                new_dimensions = opt_img.size

            # Print optimization stats
            enhancements = []
            if enhance_contrast:
                enhancements.append("contrast+")
            if convert_to_grayscale:
                enhancements.append("grayscale")

            self._print_optimization_stats(
                original_size=original_size,
                optimized_size=optimized_size,
                processing_time=processing_time,
                original_dimensions=original_dimensions,
                new_dimensions=new_dimensions,
                enhancements=enhancements
            )

            # Save to database and return fresh copy
            return self.storage_manager.database.update_image(fresh_image)

        except Exception as e:
            # Handle optimization failure - get fresh copy for error update
            fresh_image = self.storage_manager.database.get_image_by_id(image.id)
            if fresh_image:
                fresh_image.status = ProcessingStatus.FAILED
                fresh_image.processed_at = datetime.now()
                self.storage_manager.database.update_image(fresh_image)
            print(f"    Error optimizing {image.filename}: {e}")
            raise e

    def optimize_images_batch(self, images_list: List[Image] = None, max_width: int = 2048,
                              jpeg_quality: int = 90, enhance_contrast: bool = True,
                              convert_to_grayscale: bool = False) -> List[Image]:
        """
        Optimize multiple images and update database records.

        Args:
            images_list: List of images to optimize (if None, gets all pending images)
            max_width: Maximum width for resizing
            jpeg_quality: JPEG quality setting
            enhance_contrast: Whether to enhance contrast
            convert_to_grayscale: Whether to convert to grayscale

        Returns:
            List of processed Image objects
        """
        if images_list is None:
            images_list = self.storage_manager.database.get_images_by_status(ProcessingStatus.PENDING)

        if not images_list:
            print("No images to optimize")
            return []

        print(f"Optimizing {len(images_list)} images...")
        processed_images = []

        for i, image in enumerate(images_list, 1):
            print(f"  [{i}/{len(images_list)}] Processing {image.filename}...")

            try:
                optimized_image = self.optimize_image(
                    image=image,
                    max_width=max_width,
                    jpeg_quality=jpeg_quality,
                    enhance_contrast=enhance_contrast,
                    convert_to_grayscale=convert_to_grayscale
                )
                processed_images.append(optimized_image)

            except Exception as e:
                processed_images.append(image)  # Add failed image to results
                print(f"    Failed: {e}")

        successful = len([img for img in processed_images if img.status == ProcessingStatus.COMPLETED])
        print(f"\nCompleted: {successful}/{len(images_list)} images optimized successfully")

        return processed_images

    def optimize_all_pending_images(self) -> List[Image]:
        """
        Optimize all pending images using current settings.

        Returns:
            List of processed Image objects
        """
        return self.optimize_images_batch(
            images_list=None,  # Will get all pending images
            max_width=self.settings.max_width,
            jpeg_quality=self.settings.jpeg_quality,
            enhance_contrast=self.settings.enhance_contrast,
            convert_to_grayscale=self.settings.convert_to_grayscale
        )

    def optimize_specific_images(self, images_list: List[Image]) -> List[Image]:
        """
        Optimize specific images using current settings.

        Args:
            images_list: List of Image objects to optimize

        Returns:
            List of processed Image objects
        """
        return self.optimize_images_batch(
            images_list=images_list,
            max_width=self.settings.max_width,
            jpeg_quality=self.settings.jpeg_quality,
            enhance_contrast=self.settings.enhance_contrast,
            convert_to_grayscale=self.settings.convert_to_grayscale
        )

    def _create_optimized_path(self, original_path: str) -> str:
        """
        Create optimized file path with .jpg extension.

        Args:
            original_path: Original file path

        Returns:
            Optimized file path with .jpg extension
        """
        path_obj = Path(original_path)
        return str(path_obj.with_suffix('.jpg'))

    def _print_optimization_stats(self, original_size: int, optimized_size: int,
                                  processing_time: float, original_dimensions: tuple = None,
                                  new_dimensions: tuple = None, enhancements: List[str] = None) -> None:
        """
        Print detailed optimization statistics.

        Args:
            original_size: Original file size in bytes
            optimized_size: Optimized file size in bytes
            processing_time: Processing time in seconds
            original_dimensions: Original image dimensions (width, height)
            new_dimensions: New image dimensions (width, height)
            enhancements: List of applied enhancements
        """
        from data.local_storage import LocalStorage

        original_mb = LocalStorage.format_file_size(original_size)
        optimized_mb = LocalStorage.format_file_size(optimized_size)
        compression_ratio = LocalStorage.calculate_compression_ratio(original_size, optimized_size)

        size_info = f"{original_mb} → {optimized_mb}"
        if original_dimensions and new_dimensions and original_dimensions != new_dimensions:
            size_info += f" (resized from {original_dimensions})"

        enhancement_info = ""
        if enhancements:
            enhancement_info = f" [{', '.join(enhancements)}]"

        print(f"    Optimized: {size_info} (-{compression_ratio:.0f}%){enhancement_info} "
              f"in {processing_time:.1f}s")

    def run_full_pipeline(self) -> List[Image]:
        """
        Run the complete pipeline: discover, save, and optimize images.

        Returns:
            List of processed Image objects
        """
        print("🔍 Starting image processing pipeline...")

        # Step 1: Discover and save images to database
        print("\n📁 Step 1: Discovering images...")
        images = self.discover_and_save_images()

        # Step 2: Show current status
        print("\n📊 Current status:")
        self.storage_manager.print_processing_summary()

        # Step 3: Optimize pending images
        print(f"\n⚡ Step 2: Optimizing images with {self.settings.quality.value} quality...")
        processed_images = self.optimize_all_pending_images()

        # Step 4: Show final results
        print("\n✅ Pipeline completed!")
        self.storage_manager.print_processing_summary()

        return processed_images

    def search_images(self, search_term: str) -> List[Image]:
        """Search images by filename or path."""
        return self.storage_manager.search_images(search_term)

    def retry_failed_images(self) -> List[Image]:
        """
        Retry processing failed images.

        Returns:
            List of processed Image objects
        """
        failed_images = self.storage_manager.get_image(status=ProcessingStatus.FAILED)
        if not failed_images:
            print("No failed images to retry")
            return []

        print(f"🔄 Retrying {len(failed_images)} failed images...")

        # Reset status to pending
        for image in failed_images:
            self.storage_manager.update_image_status(image.id, ProcessingStatus.PENDING)

        # Retry optimization
        return self.optimize_specific_images(failed_images)

    def print_summary(self):
        """Print processing summary."""
        self.storage_manager.print_processing_summary()

    def close(self):
        """Clean up resources."""
        self.storage_manager.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for OCR optimization")
    parser.add_argument('input_path', help='Path to the folder containing images')
    parser.add_argument('--output', '-o', default='optimized_for_ocr', help='Output folder for optimized images')
    parser.add_argument('--quality', '-q', choices=['fast', 'balanced', 'best'],
                        default='balanced', help='Optimization quality preset')
    parser.add_argument('--grayscale', '-g', action='store_true',
                        help='Convert images to grayscale (good for documents)')
    parser.add_argument('--no-contrast', action='store_true',
                        help='Disable contrast enhancement')
    parser.add_argument('--discover-only', action='store_true',
                        help='Only discover and save images, do not optimize')
    parser.add_argument('--optimize-only', action='store_true',
                        help='Only optimize pending images, do not discover new ones')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry processing failed images')
    parser.add_argument('--summary', action='store_true',
                        help='Show processing summary and exit')

    args = parser.parse_args()

    # Convert quality string to enum
    quality_map = {
        'fast': OCRQuality.FAST,
        'balanced': OCRQuality.BALANCED,
        'best': OCRQuality.BEST
    }

    # Initialize processor with context manager for proper cleanup
    with ImageProcessor(
            input_folder_path=args.input_path,
            output_folder=args.output,
    ) as processor:

        # Configure optimization settings
        processor.configure_optimization(
            quality=quality_map[args.quality],
            enhance_contrast=not args.no_contrast,
            convert_to_grayscale=args.grayscale
        )

        try:
            if args.summary:
                # Just show summary
                processor.print_summary()

            elif args.retry_failed:
                # Retry failed images
                processor.retry_failed_images()

            elif args.discover_only:
                # Only discover and save images
                processor.discover_and_save_images()
                processor.print_summary()

            elif args.optimize_only:
                # Only optimize pending images
                processor.optimize_all_pending_images()
                processor.print_summary()

            else:
                # Run full pipeline
                processed_images = processor.run_full_pipeline()

                # Show results
                print(f"\n📋 Final Results:")
                for img in processed_images:
                    if img.status == ProcessingStatus.COMPLETED:
                        print(f"✓ {img.filename} -> {img.optimized_file_path}")
                    else:
                        print(f"✗ {img.filename}: {img.status.value}")

        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise