import hashlib
from pathlib import Path
from typing import List
from models.models import Image, ProcessingStatus
from image_optimizer import ImageOptimizer
from database import DatabaseManager


class BatchProcessor:
    """Discovers images, manages database state, and coordinates processing."""

    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.image_optimizer = ImageOptimizer()
        self.db_manager = DatabaseManager()

    def process_folder(self, folder_path: str) -> List[Image]:
        """Process all images in folder and return populated Image models."""
        input_path = Path(folder_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input folder '{folder_path}' does not exist")

        # Discover images
        discovered_files = self._discover_images(input_path)

        if not discovered_files:
            return []

        all_images = []
        images_to_process = []

        for file_path in discovered_files:
            # Calculate hash for each discovered image
            file_hash = self._calculate_file_hash(file_path)
            relative_path = str(file_path.relative_to(input_path))

            # Check database for existing record
            existing_image = self.db_manager.get_image_by_path(relative_path)

            if existing_image:
                # Compare hashes
                if existing_image.file_hash == file_hash:
                    # Same image - use existing data
                    all_images.append(existing_image)
                    continue
                else:
                    # Hash changed - update record and mark for processing
                    existing_image.file_hash = file_hash
                    existing_image.status = ProcessingStatus.PENDING
                    images_to_process.append(existing_image)
            else:
                # New image - create record and mark for processing
                new_image = Image(
                    file_path=relative_path,
                    optimized_file_path="",
                    filename=file_path.name,
                    file_hash=file_hash,
                    status=ProcessingStatus.PENDING
                )
                images_to_process.append(new_image)

        # Process images that need processing
        if images_to_process:
            processed_images = self.image_optimizer.optimize_images(images_to_process, input_path)

            # Update database with processed results
            for image in processed_images:
                self.db_manager.save_image(image)

            all_images.extend(processed_images)

        return all_images

    def _discover_images(self, folder_path: Path) -> List[Path]:
        """Discover images using depth-first traversal maintaining original order."""
        image_files = []

        def scan_directory(current_path: Path):
            try:
                items = sorted(current_path.iterdir(), key=lambda x: x.name)

                # Process files first
                for item in items:
                    if item.is_file() and self._is_valid_image(item):
                        image_files.append(item)

                # Then process subdirectories
                for item in items:
                    if item.is_dir():
                        scan_directory(item)

            except PermissionError:
                print(f"Warning: No permission to access {current_path}")

        scan_directory(folder_path)
        return image_files

    def _is_valid_image(self, file_path: Path) -> bool:
        """Basic validation for image files."""
        if file_path.suffix.lower() not in self.supported_formats:
            return False

        try:
            # Check if file is readable
            with open(file_path, 'rb') as f:
                f.read(1)
            return True
        except (OSError, PermissionError):
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()