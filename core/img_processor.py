import argparse
from dataclasses import dataclass
from enum import Enum
from typing import List

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

    def optimize_all_pending_images(self) -> List[Image]:
        """
        Optimize all pending images using current settings.

        Returns:
            List of processed Image objects
        """
        return self.storage_manager.optimize_images_batch(
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
        return self.storage_manager.optimize_images_batch(
            images_list=images_list,
            max_width=self.settings.max_width,
            jpeg_quality=self.settings.jpeg_quality,
            enhance_contrast=self.settings.enhance_contrast,
            convert_to_grayscale=self.settings.convert_to_grayscale
        )

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