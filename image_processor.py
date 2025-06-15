import hashlib
from pathlib import Path
from typing import List

from database import DatabaseManager


class ImageProcessor:
    """Handles image file discovery, copying, and hash calculation."""

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
        """Copy image maintaining folder structure."""
        # Calculate relative path from input folder
        try:
            relative_path = source_path.relative_to(input_folder)
        except ValueError:
            # Fallback if source is not within input folder
            relative_path = source_path.name

        destination = result_folder / "images" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Copy file if it doesn't exist or is different
        if not destination.exists():
            with open(source_path, 'rb') as src, open(destination, 'wb') as dst:
                dst.write(src.read())

        return destination
