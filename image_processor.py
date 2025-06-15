import hashlib
import os
from pathlib import Path
from typing import List

from database import DatabaseManager

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageProcessor:
    """Handles image file discovery, copying, optimization, and hash calculation."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.supported_formats = {'.jpg', '.jpeg', '.png'}

    def discover_images(self, folder_path: str) -> List[Path]:
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

        scan_directory(Path(folder_path))
        return image_files

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def copy_image(self, source_path: Path, input_folder: Path, result_folder: Path) -> Path:
        """Copy and optimize image maintaining folder structure."""
        # Calculate relative path from input folder
        try:
            relative_path = source_path.relative_to(input_folder)
        except ValueError:
            # Fallback if source is not within input folder
            relative_path = source_path.name

        # Always save optimized images as .jpg for consistency and smaller size
        destination = result_folder / "images" / relative_path
        if destination.suffix.lower() in {'.png'}:
            destination = destination.with_suffix('.jpg')

        destination.parent.mkdir(parents=True, exist_ok=True)

        # Optimize and copy if it doesn't exist
        if not destination.exists():
            if HAS_PIL:
                self._optimize_and_save(source_path, destination)
            else:
                self._simple_copy(source_path, destination)

        return destination

    def _optimize_and_save(self, source_path: Path, destination: Path):
        """Optimize image during copying for faster Vision API processing."""
        import time

        start_time = time.time()
        original_size = source_path.stat().st_size

        try:
            with PILImage.open(source_path) as img:
                # Convert to RGB if needed (remove transparency, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    # Create white background for transparent images
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if 'transparency' in img.info:
                        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if image is too large
                # Vision API optimal: 1024x1024 to 2048x2048 pixels
                max_dimension = 1600  # Good balance between quality and speed
                original_dimensions = img.size
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                # Save as optimized JPEG
                img.save(destination, format='JPEG', quality=85, optimize=True)

            # Report optimization results
            optimized_size = destination.stat().st_size
            original_mb = original_size / (1024 * 1024)
            optimized_mb = optimized_size / (1024 * 1024)
            compression_ratio = (1 - optimized_mb / original_mb) * 100 if original_mb > 0 else 0
            processing_time = time.time() - start_time

            size_info = f"{original_mb:.1f}MB → {optimized_mb:.1f}MB"
            if original_dimensions != img.size:
                size_info += f" (resized from {original_dimensions})"

            print(f"    Optimized: {size_info} (-{compression_ratio:.0f}%) in {processing_time:.1f}s")

        except Exception as e:
            print(f"    Warning: Could not optimize {source_path.name}, copying original: {e}")
            self._simple_copy(source_path, destination)

    def _simple_copy(self, source_path: Path, destination: Path):
        """Simple file copy fallback when PIL is not available or optimization fails."""
        original_size = source_path.stat().st_size
        with open(source_path, 'rb') as src, open(destination, 'wb') as dst:
            dst.write(src.read())

        if not HAS_PIL:
            size_mb = original_size / (1024 * 1024)
            print(f"    Copied: {size_mb:.1f}MB (PIL not available - no optimization)")
        else:
            print(f"    Copied original (optimization failed)")