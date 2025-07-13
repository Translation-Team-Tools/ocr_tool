import hashlib
import time
from io import BytesIO
from typing import List, Tuple

from PIL import Image as PILImage, ImageEnhance
from fs import open_fs

supported_formats = {'.jpg', '.jpeg', '.png'}


class LocalStorage:

    def __init__(self, base_path: str):
        """
        Initialize LocalStorage with a base path.

        Args:
            base_path: Base directory for all file operations
        """
        self.base_path = base_path
        self.fs = open_fs(base_path, create=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the filesystem connection."""
        if hasattr(self, 'fs') and self.fs:
            self.fs.close()

    def discover_files_recursive(self, subfolder: str = '/') -> List[str]:
        """
        Discover supported files recursively in folder with depth-first traversal.

        Args:
            subfolder: Subfolder within base path to scan (default: root)

        Returns:
            List of relative file paths in filesystem order
        """
        files = []

        def scan_directory(current_path: str):
            try:
                # Get sorted items
                items = sorted(self.fs.listdir(current_path))

                # Process files first
                for item in items:
                    item_path = self.fs.join(current_path, item)
                    if self.fs.isfile(item_path):
                        _, ext = self.fs.splitext(item)
                        if ext.lower() in supported_formats:
                            files.append(item_path)

                # Then process subdirectories
                for item in items:
                    item_path = self.fs.join(current_path, item)
                    if self.fs.isdir(item_path):
                        scan_directory(item_path)

            except Exception as e:
                print(f"Warning: Cannot access {current_path}: {e}")

        scan_directory(subfolder)
        return files

    def ensure_directory_exists(self, path: str) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            path: Directory path relative to base path
        """
        dir_path = self.fs.dirname(path)
        if dir_path and not self.fs.exists(dir_path):
            self.fs.makedirs(dir_path, recreate=True)

    def get_file_size(self, path: str) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path relative to base path

        Returns:
            File size in bytes
        """
        return self.fs.getsize(path)

    def file_exists(self, path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: File path relative to base path

        Returns:
            True if file exists
        """
        return self.fs.exists(path)

    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file.

        Args:
            file_path: Path to file relative to base path

        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()

        with self.fs.open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def save_optimized_image(self, image: PILImage.Image, destination: str,
                             quality: int = 90) -> Tuple[int, int, float]:
        """
        Save optimized image and return processing stats.

        Args:
            image: PIL Image object to save
            destination: Destination path relative to base path
            quality: JPEG quality (1-100)

        Returns:
            Tuple of (original_size, optimized_size, processing_time)
        """
        start_time = time.time()

        # Ensure destination directory exists
        self.ensure_directory_exists(destination)

        # Save as optimized JPEG
        with self.fs.open(destination, 'wb') as f:
            image.save(f, format='JPEG', quality=quality, optimize=True)

        processing_time = time.time() - start_time
        optimized_size = self.get_file_size(destination)

        return 0, optimized_size, processing_time

    def optimize_image_for_ocr(self, source_path: str, destination: str,
                               max_width: int = 2048, jpeg_quality: int = 90,
                               enhance_contrast: bool = True,
                               convert_to_grayscale: bool = False) -> Tuple[int, int, float]:
        """
        Complete image optimization pipeline for OCR.

        Args:
            source_path: Source image path relative to base path
            destination: Destination path relative to base path
            max_width: Maximum width for resizing
            jpeg_quality: JPEG quality setting
            enhance_contrast: Whether to enhance contrast
            convert_to_grayscale: Whether to convert to grayscale

        Returns:
            Tuple of (original_size, optimized_size, processing_time)
        """
        start_time = time.time()
        original_size = self.get_file_size(source_path)

        try:
            # Read image data
            with self.fs.open(source_path, 'rb') as f:
                img_obj = PILImage.open(BytesIO(f.read()))

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
                self.ensure_directory_exists(destination)

                with self.fs.open(destination, 'wb') as dest_f:
                    img_obj.save(dest_f, format='JPEG',
                                 quality=jpeg_quality, optimize=True)

            processing_time = time.time() - start_time
            optimized_size = self.get_file_size(destination)

            return original_size, optimized_size, processing_time

        except Exception as e:
            # Fallback to simple copy
            self.ensure_directory_exists(destination)
            self.fs.copy(source_path, destination)
            processing_time = time.time() - start_time
            return original_size, original_size, processing_time

    def copy_file(self, source_path: str, destination: str) -> None:
        """
        Copy file within the storage.

        Args:
            source_path: Source file path relative to base path
            destination: Destination file path relative to base path
        """
        self.ensure_directory_exists(destination)
        self.fs.copy(source_path, destination)

    def delete_file(self, path: str) -> None:
        """
        Delete file from storage.

        Args:
            path: File path relative to base path
        """
        if self.fs.exists(path):
            self.fs.remove(path)

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Get absolute system path for a relative path.

        Args:
            relative_path: Path relative to base path

        Returns:
            Absolute system path
        """
        return self.fs.getsyspath(relative_path)

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