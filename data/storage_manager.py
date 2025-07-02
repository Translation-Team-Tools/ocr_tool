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
        self.input_folder_path = Path(input_folder_path)

        # Place output folder inside results directory
        if project_root:
            project_root_path = Path(project_root)
        else:
            project_root_path = Path.cwd()

        self.output_folder = project_root_path / "result" / output_folder

        # Initialize file system handler
        self.local_storage = LocalStorage()

        # Initialize database handler
        self.database = Database(input_folder_path, project_root)

        # Validate input path exists
        self.local_storage.validate_path_exists(str(self.input_folder_path))

    def discover_and_save_images(self) -> List[Image]:
        """
        Discover all images in input folder and save them to database.

        Returns:
            List of Image objects saved to database
        """
        print(f"Discovering images in {self.input_folder_path}...")

        # Discover image files using LocalStorage
        image_files = self.local_storage.discover_files_recursive(self.input_folder_path)
        images_list = []

        print(f"Found {len(image_files)} images")

        for file_path in image_files:
            try:
                # Calculate relative path
                relative_path = self.local_storage.get_relative_path(file_path, self.input_folder_path)

                # Calculate file hash
                file_hash = self.local_storage.calculate_file_hash(file_path)

                # Check if image already exists in database
                existing_image = self.database.get_image_by_hash(file_hash)
                if existing_image:
                    print(f"  Skipped {file_path.name} (already in database)")
                    images_list.append(existing_image)
                    continue

                # Create new Image object
                image = Image(
                    filename=file_path.name,
                    original_file_path=str(relative_path),
                    file_hash=file_hash,
                    status=ProcessingStatus.PENDING,
                    optimized_file_path="",  # Will be set during optimization
                    vision_json_path=""  # Will be set during OCR processing
                )

                # Save to database
                saved_image = self.database.add_image(image)
                images_list.append(saved_image)
                print(f"  Added {file_path.name} to database")

            except Exception as e:
                print(f"Warning: Could not process {file_path.name}: {e}")

        print(f"Database contains {len(images_list)} images")
        return images_list

    def optimize_image(self, image: Image, max_width: int = 2048, jpeg_quality: int = 90,
                       enhance_contrast: bool = True, convert_to_grayscale: bool = False) -> Image:
        """
        Optimize a single image and update database record.
        """
        try:
            # Get fresh copy from database to avoid session issues
            fresh_image = self.database.get_image_by_id(image.id)
            if not fresh_image:
                raise ValueError(f"Image with ID {image.id} not found in database")

            # Update status to processing
            fresh_image.status = ProcessingStatus.PROCESSING
            fresh_image = self.database.update_image(fresh_image)

            # Construct source path
            source_path = self.input_folder_path / fresh_image.original_file_path

            # Create optimized output path
            optimized_path = self.local_storage.create_output_path(
                str(self.output_folder),
                fresh_image.original_file_path,
                force_extension='.jpg'
            )

            # Check if already optimized
            if self.local_storage.file_exists(optimized_path):
                print(f"    Skipped: {fresh_image.filename} already optimized")
                # Update database with existing path
                fresh_image.optimized_file_path = str(
                    self.local_storage.get_relative_path(optimized_path, self.output_folder))
                fresh_image.status = ProcessingStatus.COMPLETED
                fresh_image.processed_at = datetime.now()
                return self.database.update_image(fresh_image)

            # Optimize image using LocalStorage
            original_size, optimized_size, processing_time = self.local_storage.optimize_image_for_ocr(
                source_path=source_path,
                destination=optimized_path,
                max_width=max_width,
                jpeg_quality=jpeg_quality,
                enhance_contrast=enhance_contrast,
                convert_to_grayscale=convert_to_grayscale
            )

            # Update Image object with results
            fresh_image.optimized_file_path = str(
                self.local_storage.get_relative_path(optimized_path, self.output_folder))
            fresh_image.status = ProcessingStatus.COMPLETED
            fresh_image.processed_at = datetime.now()

            # Print optimization stats
            with self.local_storage.open_image(source_path) as orig_img:
                original_dimensions = orig_img.size
            with self.local_storage.open_image(optimized_path) as opt_img:
                new_dimensions = opt_img.size

            enhancements = []
            if enhance_contrast:
                enhancements.append("contrast+")
            if convert_to_grayscale:
                enhancements.append("grayscale")

            self.local_storage.print_optimization_stats(
                original_size=original_size,
                optimized_size=optimized_size,
                processing_time=processing_time,
                original_dimensions=original_dimensions,
                new_dimensions=new_dimensions,
                enhancements=enhancements
            )

            # Save to database and return fresh copy
            return self.database.update_image(fresh_image)

        except Exception as e:
            # Handle optimization failure - get fresh copy for error update
            fresh_image = self.database.get_image_by_id(image.id)
            if fresh_image:
                fresh_image.status = ProcessingStatus.FAILED
                fresh_image.processed_at = datetime.now()
                self.database.update_image(fresh_image)
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
            images_list = self.database.get_image(status=ProcessingStatus.PENDING)

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

    def get_image(self, image_id: int = None, file_hash: str = None, filename: str = None, status: ProcessingStatus = None):
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
    def get_optimized_file_path(self, image: Image) -> Optional[Path]:
        """
        Get full path to optimized file.

        Args:
            image: Image object

        Returns:
            Path to optimized file or None if not optimized
        """
        if not image.optimized_file_path:
            return None
        return self.output_folder / image.optimized_file_path

    def get_original_file_path(self, image: Image) -> Path:
        """
        Get full path to original file.

        Args:
            image: Image object

        Returns:
            Path to original file
        """
        return self.input_folder_path / image.original_file_path

    def optimized_file_exists(self, image: Image) -> bool:
        """
        Check if optimized file exists on disk.

        Args:
            image: Image object

        Returns:
            True if optimized file exists
        """
        optimized_path = self.get_optimized_file_path(image)
        return optimized_path and self.local_storage.file_exists(optimized_path)

    def original_file_exists(self, image: Image) -> bool:
        """
        Check if original file exists on disk.

        Args:
            image: Image object

        Returns:
            True if original file exists
        """
        original_path = self.get_original_file_path(image)
        return self.local_storage.file_exists(original_path)

    def close(self):
        """Clean up resources."""
        self.database.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()