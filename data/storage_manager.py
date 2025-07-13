from datetime import datetime
from pathlib import Path
from typing import List, Optional

from data.database import Database
from data.local_storage import LocalStorage
from data.models import Image, ProcessingStatus


class StorageManager:
    """
    Unified storage manager that handles both file system operations and database operations for images.
    Provides a clean interface for image processing workflows.
    """

    def __init__(self, input_folder_path: str, output_folder: str = "optimized_images", project_root: str = None):
        """
        Initialize StorageManager with both file system and database handling.

        Args:
            input_folder_path: Path to input folder containing images
            output_folder: Where to save optimized images (will be placed in results/ folder)
            project_root: Project root for database location
        """
        self.input_folder_path = input_folder_path

        # Place output folder inside results directory
        if project_root:
            project_root_path = Path(project_root)
        else:
            project_root_path = Path.cwd()

        self.output_folder_path = str(project_root_path / "result" / output_folder)

        # Initialize file system handlers
        self.input_storage = LocalStorage(input_folder_path)
        self.output_storage = LocalStorage(self.output_folder_path)

        # Initialize database handler
        self.database = Database(input_folder_path, project_root)

    def discover_and_save_images(self) -> List[Image]:
        """
        Discover all images in input folder and save them to database.

        Returns:
            List of Image objects saved to database
        """
        print(f"Discovering images in {self.input_folder_path}...")

        # Discover image files using LocalStorage
        image_files = self.input_storage.discover_files_recursive()
        images_list = []

        print(f"Found {len(image_files)} images")

        for relative_file_path in image_files:
            try:
                # Get filename from path
                filename = Path(relative_file_path).name

                # Calculate file hash
                file_hash = self.input_storage.calculate_file_hash(relative_file_path)

                # Check if image already exists in database
                existing_image = self.database.get_image_by_hash(file_hash)
                if existing_image:
                    print(f"  Skipped {filename} (already in database)")
                    images_list.append(existing_image)
                    continue

                # Create new Image object
                image = Image(
                    filename=filename,
                    original_file_path=relative_file_path,
                    file_hash=file_hash,
                    status=ProcessingStatus.PENDING,
                    optimized_file_path="",  # Will be set during optimization
                    vision_json_path=""  # Will be set during OCR processing
                )

                # Save to database
                saved_image = self.database.add_image(image)
                images_list.append(saved_image)
                print(f"  Added {filename} to database")

            except Exception as e:
                filename = Path(relative_file_path).name if relative_file_path else "unknown"
                print(f"Warning: Could not process {filename}: {e}")

        print(f"Database contains {len(images_list)} images")
        return images_list

    def get_image(self, image_id: int = None, file_hash: str = None, filename: str = None,
                  status: ProcessingStatus = None):
        if image_id: return self.database.get_image_by_id(image_id=image_id)
        if file_hash: return self.database.get_image_by_hash(file_hash=file_hash)
        if filename: return self.database.get_image_by_filename(filename=filename)
        if status: return self.database.get_images_by_status(status=status)
        return False

    def get_all_images(self) -> List[Image]:
        """Get all images."""
        return self.database.get_all_images()

    def search_images(self, search_term: str) -> List[Image]:
        """Search images by filename or path."""
        return self.database.search_images(search_term)

    def update_image_status(self, image_id: int, status: ProcessingStatus) -> Optional[Image]:
        """
        Update image status.

        Args:
            image_id: ID of image to update
            status: New status

        Returns:
            Updated Image object or None if not found
        """
        image = self.get_image(image_id=image_id)
        if image:
            image.status = status
            if status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                image.processed_at = datetime.now()
            return self.database.update_image(image)
        return None

    def delete_image(self, image_id: int) -> bool:
        """
        Delete image from database.

        Args:
            image_id: ID of image to delete

        Returns:
            True if deleted, False if not found
        """
        return self.database.delete_image(image_id)

    def get_processing_summary(self) -> dict:
        """
        Get summary of processing status.

        Returns:
            Dictionary with counts by status
        """
        all_images = self.get_all_images()
        summary = {
            'total': len(all_images),
            'pending': len([img for img in all_images if img.status == ProcessingStatus.PENDING]),
            'processing': len([img for img in all_images if img.status == ProcessingStatus.PROCESSING]),
            'completed': len([img for img in all_images if img.status == ProcessingStatus.COMPLETED]),
            'failed': len([img for img in all_images if img.status == ProcessingStatus.FAILED]),
            'skipped': len([img for img in all_images if img.status == ProcessingStatus.SKIPPED])
        }
        return summary

    def print_processing_summary(self):
        """Print a formatted processing summary."""
        summary = self.get_processing_summary()
        print(f"\n📊 Processing Summary:")
        print(f"   Total images: {summary['total']}")
        print(f"   Pending: {summary['pending']}")
        print(f"   Processing: {summary['processing']}")
        print(f"   Completed: {summary['completed']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Skipped: {summary['skipped']}")

    # File system utility methods
    def get_optimized_file_path(self, image: Image) -> Optional[str]:
        """
        Get full absolute path to optimized file.

        Args:
            image: Image object

        Returns:
            Absolute path to optimized file or None if not optimized
        """
        if not image.optimized_file_path:
            return None
        return self.output_storage.get_absolute_path(image.optimized_file_path)

    def get_original_file_path(self, image: Image) -> str:
        """
        Get full absolute path to original file.

        Args:
            image: Image object

        Returns:
            Absolute path to original file
        """
        return self.input_storage.get_absolute_path(image.original_file_path)

    def optimized_file_exists(self, image: Image) -> bool:
        """
        Check if optimized file exists on disk.

        Args:
            image: Image object

        Returns:
            True if optimized file exists
        """
        if not image.optimized_file_path:
            return False
        return self.output_storage.file_exists(image.optimized_file_path)

    def original_file_exists(self, image: Image) -> bool:
        """
        Check if original file exists on disk.

        Args:
            image: Image object

        Returns:
            True if original file exists
        """
        return self.input_storage.file_exists(image.original_file_path)

    def close(self):
        """Clean up resources."""
        self.database.close()
        self.input_storage.close()
        self.output_storage.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()