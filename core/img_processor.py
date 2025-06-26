import argparse
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from PIL import Image as PILImage, ImageEnhance

from data.models import Image, ProcessingStatus


class OCRQuality(Enum):
    """Simple quality options for OCR optimization."""
    FAST = "fast"  # Smaller files, faster processing
    BALANCED = "balanced"  # Good balance of quality and speed
    BEST = "best"  # Highest quality for difficult text


@dataclass
class OptimizationSettings:
    """OCR optimization settings with user-friendly options."""
    quality: OCRQuality = OCRQuality.BALANCED
    enhance_contrast: bool = True  # Makes text more readable
    convert_to_grayscale: bool = False  # Can improve OCR for some documents
    max_width: int = 2048  # Larger images for better text recognition
    jpeg_quality: int = 90  # High quality for text preservation

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

    Features:
    - Discovers images in folders recursively
    - Creates structured Image objects with metadata
    - Optimizes images specifically for text recognition
    - Provides simple quality presets for non-technical users
    - Tracks processing status and errors
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
        input_path = Path(input_folder_path)
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder_path}")

        image_files = self._discover_images(input_path)
        images_list = []

        print(f"Found {len(image_files)} images in {input_folder_path}")

        for file_path in image_files:
            try:
                # Calculate relative path for organization
                relative_path = file_path.relative_to(input_path)

                # Create Image object
                image = Image(
                    filename=file_path.name,
                    original_file_path=str(relative_path),
                    file_hash=self._calculate_file_hash(file_path),
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

                # Optimize and save image
                optimized_path = self._optimize_and_save_image(source_path, image.original_file_path)

                # Update image model
                image.optimized_file_path = str(optimized_path.relative_to(self.output_folder))
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

    def _discover_images(self, folder_path: Path) -> List[Path]:
        """Discover images using depth-first traversal maintaining original order."""
        image_files = []

        def scan_directory(current_path: Path):
            try:
                # Get all items and sort them as they appear in filesystem
                items = sorted(current_path.iterdir(), key=lambda x: x.name)

                # First, process files in current directory
                for item in items:
                    if item.is_file() and item.suffix.lower() in self.supported_formats:
                        image_files.append(item)

                # Then, recursively process subdirectories
                for item in items:
                    if item.is_dir():
                        scan_directory(item)

            except PermissionError:
                print(f"Warning: No permission to access {current_path}")

        scan_directory(folder_path)
        return image_files

    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _optimize_and_save_image(self, source_path: Path, relative_path: str) -> Path:
        """Optimize single image for OCR and save."""
        # Always save as .jpg for consistency and optimal OCR processing
        destination = self.output_folder / relative_path
        if destination.suffix.lower() == '.png':
            destination = destination.with_suffix('.jpg')

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if destination.exists():
            print(f"    Skipped: Already exists")
            return destination

        # Optimize and save
        self._ocr_optimize_and_save(source_path, destination)
        return destination

    def _ocr_optimize_and_save(self, source_path: Path, destination: Path):
        """Optimize image specifically for OCR processing."""
        start_time = time.time()
        original_size = source_path.stat().st_size

        try:
            with PILImage.open(source_path) as img_obj:
                # Convert to RGB first
                if img_obj.mode in ('RGBA', 'P', 'LA'):
                    # Create white background for transparent images (better for OCR)
                    background = PILImage.new('RGB', img_obj.size, (255, 255, 255))
                    if img_obj.mode == 'P':
                        img_obj = img_obj.convert('RGBA')
                    if 'transparency' in img_obj.info:
                        background.paste(img_obj, mask=img_obj.split()[-1])
                    else:
                        background.paste(img_obj)
                    img_obj = background
                elif img_obj.mode != 'RGB':
                    img_obj = img_obj.convert('RGB')

                # Convert to grayscale if requested (can improve OCR for documents)
                if self.settings.convert_to_grayscale:
                    img_obj = img_obj.convert('L').convert('RGB')  # Convert back to RGB for JPEG

                # Enhance contrast for better text recognition
                if self.settings.enhance_contrast:
                    enhancer = ImageEnhance.Contrast(img_obj)
                    img_obj = enhancer.enhance(1.2)  # Slight contrast boost

                # Resize for optimal OCR processing
                original_dimensions = img_obj.size
                if max(img_obj.size) > self.settings.max_width:
                    ratio = self.settings.max_width / max(img_obj.size)
                    new_size = tuple(int(dim * ratio) for dim in img_obj.size)
                    img_obj = img_obj.resize(new_size, PILImage.Resampling.LANCZOS)

                # Save as optimized JPEG with high quality for text preservation
                img_obj.save(destination, format='JPEG',
                         quality=self.settings.jpeg_quality,
                         optimize=True)

            # Report optimization results
            optimized_size = destination.stat().st_size
            original_mb = original_size / (1024 * 1024)
            optimized_mb = optimized_size / (1024 * 1024)
            compression_ratio = (1 - optimized_mb / original_mb) * 100 if original_mb > 0 else 0
            processing_time = time.time() - start_time

            size_info = f"{original_mb:.1f}MB → {optimized_mb:.1f}MB"
            if original_dimensions != img_obj.size:
                size_info += f" (resized from {original_dimensions})"

            enhancements = []
            if self.settings.enhance_contrast:
                enhancements.append("contrast+")
            if self.settings.convert_to_grayscale:
                enhancements.append("grayscale")

            enhancement_info = f" [{', '.join(enhancements)}]" if enhancements else ""

            print(f"    Optimized: {size_info} (-{compression_ratio:.0f}%){enhancement_info} "
                  f"in {processing_time:.1f}s")

        except Exception as e:
            print(f"    Warning: Could not optimize {source_path.name}, copying original: {e}")
            self._simple_copy(source_path, destination)

    @staticmethod
    def _simple_copy(source_path: Path, destination: Path):
        """Simple file copy fallback."""
        original_size = source_path.stat().st_size
        with open(source_path, 'rb') as src, open(destination, 'wb') as dst:
            dst.write(src.read())

        size_mb = original_size / (1024 * 1024)
        print(f"    Copied: {size_mb:.1f}MB (no optimization)")


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