import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image as PILImage, ImageEnhance

supported_formats = {'.jpg', '.jpeg', '.png'}


class LocalStorage:
    """
    Handles all local file system operations for image processing.
    Provides a clean interface for file and directory manipulations.
    """

    # Path Operations
    @staticmethod
    def validate_path_exists(path_str: str) -> Path:
        """
        Validate that a path exists and return Path object.

        Args:
            path_str: String path to validate

        Returns:
            Path object

        Raises:
            ValueError: If path doesn't exist
        """
        path = Path(path_str)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path_str}")
        return path

    @staticmethod
    def get_relative_path(file_path: Path, base_path: Path) -> Path:
        """
        Calculate relative path from base path.

        Args:
            file_path: Full file path
            base_path: Base path to calculate relative to

        Returns:
            Relative path
        """
        return file_path.relative_to(base_path)

    @staticmethod
    def create_output_path(output_folder: str, relative_path: str,
                           force_extension: str = None) -> Path:
        """
        Create output path with optional extension change.

        Args:
            output_folder: Base output folder
            relative_path: Relative path from input
            force_extension: Optional extension to force (e.g., '.jpg')

        Returns:
            Output path
        """
        destination = Path(output_folder) / relative_path
        if force_extension and destination.suffix.lower() != force_extension.lower():
            destination = destination.with_suffix(force_extension)
        return destination

    # Directory Operations
    @staticmethod
    def ensure_directory_exists(path: Path) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            path: Directory path to create
        """
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def discover_files_recursive(folder_path: Path) -> List[Path]:
        """
        Discover supported files recursively in folder with depth-first traversal.

        Args:
            folder_path: Root folder to scan

        Returns:
            List of file paths in filesystem order
        """
        files = []

        def scan_directory(current_path: Path):
            try:
                # Get sorted items
                items = sorted(current_path.iterdir(), key=lambda x: x.name)

                # Process files first
                for item in items:
                    if item.is_file() and item.suffix.lower() in supported_formats:
                        files.append(item)

                # Then process subdirectories
                for item in items:
                    if item.is_dir():
                        scan_directory(item)

            except PermissionError:
                print(f"Warning: No permission to access {current_path}")

        scan_directory(folder_path)
        return files

    # File Operations
    @staticmethod
    def file_exists(path: Path) -> bool:
        """
        Check if file exists.

        Args:
            path: File path to check

        Returns:
            True if file exists
        """
        return path.exists()

    @staticmethod
    def get_file_size(path: Path) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path

        Returns:
            File size in bytes
        """
        return path.stat().st_size

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    @staticmethod
    def copy_file(source_path: Path, destination: Path) -> None:
        """
        Simple file copy operation.

        Args:
            source_path: Source file path
            destination: Destination file path
        """
        # Ensure destination directory exists
        LocalStorage.ensure_directory_exists(destination.parent)

        with open(source_path, 'rb') as src, open(destination, 'wb') as dst:
            dst.write(src.read())

    # Image Operations
    @staticmethod
    def open_image(path: Path) -> PILImage.Image:
        """
        Open image file using PIL.

        Args:
            path: Image file path

        Returns:
            PIL Image object
        """
        return PILImage.open(path)

    @staticmethod
    def save_optimized_image(image: PILImage.Image, destination: Path,
                             quality: int = 90) -> Tuple[int, int, float]:
        """
        Save optimized image and return processing stats.

        Args:
            image: PIL Image object to save
            destination: Destination path
            quality: JPEG quality (1-100)

        Returns:
            Tuple of (original_size, optimized_size, processing_time)
        """
        start_time = time.time()

        # Ensure destination directory exists
        LocalStorage.ensure_directory_exists(destination.parent)

        # Save as optimized JPEG
        image.save(destination, format='JPEG', quality=quality, optimize=True)

        processing_time = time.time() - start_time
        optimized_size = LocalStorage.get_file_size(destination)

        return 0, optimized_size, processing_time  # Original size would need to be passed separately

    @staticmethod
    def optimize_image_for_ocr(source_path: Path, destination: Path,
                               max_width: int = 2048, jpeg_quality: int = 90,
                               enhance_contrast: bool = True,
                               convert_to_grayscale: bool = False) -> Tuple[int, int, float]:
        """
        Complete image optimization pipeline for OCR.

        Args:
            source_path: Source image path
            destination: Destination path
            max_width: Maximum width for resizing
            jpeg_quality: JPEG quality setting
            enhance_contrast: Whether to enhance contrast
            convert_to_grayscale: Whether to convert to grayscale

        Returns:
            Tuple of (original_size, optimized_size, processing_time)
        """
        start_time = time.time()
        original_size = LocalStorage.get_file_size(source_path)

        try:
            with LocalStorage.open_image(source_path) as img_obj:
                # Convert to RGB first
                if img_obj.mode in ('RGBA', 'P', 'LA'):
                    # Create white background for transparent images
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

                # Convert to grayscale if requested
                if convert_to_grayscale:
                    img_obj = img_obj.convert('L').convert('RGB')

                # Enhance contrast for better text recognition
                if enhance_contrast:
                    enhancer = ImageEnhance.Contrast(img_obj)
                    img_obj = enhancer.enhance(1.2)

                # Resize for optimal OCR processing
                if max(img_obj.size) > max_width:
                    ratio = max_width / max(img_obj.size)
                    new_size = tuple(int(dim * ratio) for dim in img_obj.size)
                    img_obj = img_obj.resize(new_size, PILImage.Resampling.LANCZOS)

                # Save optimized image
                LocalStorage.ensure_directory_exists(destination.parent)
                img_obj.save(destination, format='JPEG',
                             quality=jpeg_quality, optimize=True)

            processing_time = time.time() - start_time
            optimized_size = LocalStorage.get_file_size(destination)

            return original_size, optimized_size, processing_time

        except Exception as e:
            # Fallback to simple copy
            LocalStorage.copy_file(source_path, destination)
            processing_time = time.time() - start_time
            return original_size, original_size, processing_time

    # Utility Methods
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string (e.g., "1.5MB")
        """
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.1f}MB"

    @staticmethod
    def calculate_compression_ratio(original_size: int, optimized_size: int) -> float:
        """
        Calculate compression ratio as percentage.

        Args:
            original_size: Original file size in bytes
            optimized_size: Optimized file size in bytes

        Returns:
            Compression ratio as percentage (0-100)
        """
        if original_size == 0:
            return 0
        return (1 - optimized_size / original_size) * 100

    @staticmethod
    def print_optimization_stats(original_size: int, optimized_size: int,
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