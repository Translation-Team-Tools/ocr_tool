import time
from pathlib import Path
from typing import List
from models import Image, ProcessingStatus

try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageOptimizer:
    """Handles image copying and optimization."""

    def __init__(self, result_folder: str = "result"):
        self.result_folder = Path(result_folder)
        self.max_dimension = 1600  # Optimal for Vision API

    def optimize_images(self, images: List[Image], input_folder: Path) -> List[Image]:
        """Copy and optimize images, return updated Image models."""
        processed_images = []

        for image in images:
            try:
                # Construct full source path
                source_path = input_folder / image.file_path

                # Copy and optimize image
                optimized_path = self._copy_and_optimize(source_path, image.file_path)

                # Update image model
                image.optimized_file_path = str(optimized_path)
                image.status = ProcessingStatus.COMPLETED

                processed_images.append(image)

            except Exception as e:
                # Handle processing errors
                image.status = ProcessingStatus.FAILED
                image.error_message = str(e)
                processed_images.append(image)
                print(f"Error processing {image.filename}: {e}")

        return processed_images

    def _copy_and_optimize(self, source_path: Path, relative_path: str) -> Path:
        """Copy and optimize single image."""
        # Always save optimized images as .jpg for consistency
        destination = self.result_folder / "images" / relative_path
        if destination.suffix.lower() == '.png':
            destination = destination.with_suffix('.jpg')

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if destination.exists():
            return destination

        # Optimize and copy
        if HAS_PIL:
            self._optimize_and_save(source_path, destination)
        else:
            self._simple_copy(source_path, destination)

        return destination

    def _optimize_and_save(self, source_path: Path, destination: Path):
        """Optimize image during copying for faster Vision API processing."""
        start_time = time.time()
        original_size = source_path.stat().st_size

        try:
            with PILImage.open(source_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P', 'LA'):
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if 'transparency' in img.info:
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large
                original_dimensions = img.size
                if max(img.size) > self.max_dimension:
                    ratio = self.max_dimension / max(img.size)
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
        """Simple file copy fallback."""
        original_size = source_path.stat().st_size
        with open(source_path, 'rb') as src, open(destination, 'wb') as dst:
            dst.write(src.read())

        if not HAS_PIL:
            size_mb = original_size / (1024 * 1024)
            print(f"    Copied: {size_mb:.1f}MB (PIL not available - no optimization)")
        else:
            print(f"    Copied original (optimization failed)")