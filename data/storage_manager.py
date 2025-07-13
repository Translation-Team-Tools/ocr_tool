from datetime import datetime
from pathlib import Path
from typing import List, Optional

from data.database import Database
from data.local_storage import LocalStorage
from data.models import Image, ImageModel, ProcessingStatus


class StorageManager:
    """Simple CRUD interface for images using database and file storage."""

    def __init__(self, input_folder_path: str, output_folder: str = "optimized_images", project_root: str = None):
        self.input_folder_path = input_folder_path

        if project_root:
            project_root_path = Path(project_root)
        else:
            project_root_path = Path.cwd()

        self.output_folder_path = str(project_root_path / "result" / output_folder)

        self.input_storage = LocalStorage(input_folder_path)
        self.output_storage = LocalStorage(self.output_folder_path)
        self.database = Database(input_folder_path, project_root)

    def _load_image_bytes(self, image_model: ImageModel) -> bytes:
        """Load optimized bytes if available, otherwise original bytes."""
        if (image_model.optimized_file_path and
                image_model.optimized_file_path != "" and
                self.output_storage.file_exists(image_model.optimized_file_path)):
            return self.output_storage.read_file_bytes(image_model.optimized_file_path)
        return self.input_storage.read_file_bytes(image_model.original_file_path)

    def _create_image_from_model(self, image_model: ImageModel) -> Image:
        """Create Image object with loaded bytes."""
        image_bytes = self._load_image_bytes(image_model)
        return Image(
            image_bytes=image_bytes,
            image_model=image_model,
            vision_response=None,
            analysis_results=None
        )

    def discover_images(self) -> List[Image]:
        """Discover image files and create database records."""
        image_files = self.input_storage.discover_files_recursive()
        images_list = []

        for relative_file_path in image_files:
            try:
                filename = Path(relative_file_path).name
                file_hash = self.input_storage.calculate_file_hash(relative_file_path)

                # Check if already exists
                existing_model = self.database.get_image_by_hash(file_hash)
                if existing_model:
                    images_list.append(self._create_image_from_model(existing_model))
                    continue

                # Create new database record
                image_model = ImageModel(
                    filename=filename,
                    original_file_path=relative_file_path,
                    file_hash=file_hash,
                    status=ProcessingStatus.PENDING
                )

                saved_model = self.database.add_image(image_model)
                images_list.append(self._create_image_from_model(saved_model))

            except Exception as e:
                print(f"Warning: Could not process {relative_file_path}: {e}")

        return images_list

    def get_image(self, image_id: int = None, file_hash: str = None,
                  filename: str = None, status: ProcessingStatus = None) -> Optional[Image]:
        """Get single image by various criteria."""
        model = self.database.get_image(image_id, file_hash, filename, status)
        if isinstance(model, list) and model:
            model = model[0]  # Take first for status queries
        return self._create_image_from_model(model) if model else None

    def get_all_images(self) -> List[Image]:
        """Get all images."""
        models = self.database.get_all_images()
        return [self._create_image_from_model(model) for model in models]

    def get_images_by_status(self, status: ProcessingStatus) -> List[Image]:
        """Get images by status."""
        models = self.database.get_images_by_status(status)
        return [self._create_image_from_model(model) for model in models]

    def search_images(self, search_term: str) -> List[Image]:
        """Search images."""
        models = self.database.search_images(search_term)
        return [self._create_image_from_model(model) for model in models]

    def update_image(self, image: Image) -> Image:
        """Update image in database."""
        if not image.image_model:
            raise ValueError("Image must have image_model")

        if image.image_model.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            image.image_model.processed_at = datetime.now()

        updated_model = self.database.update_image(image.image_model)
        return Image(
            image_bytes=image.image_bytes,
            vision_response=image.vision_response,
            analysis_results=image.analysis_results,
            image_model=updated_model
        )

    def update_status(self, image: Image, status: ProcessingStatus) -> Image:
        """Update image status."""
        image.image_model.status = status
        return self.update_image(image)

    def save_image(self, image: Image, image_bytes: bytes, filename: str = None) -> Image:
        """Save image bytes to storage and update database."""
        if not image.image_model:
            raise ValueError("Image must have image_model")

        save_filename = filename or image.image_model.filename
        self.output_storage.write_file_bytes(save_filename, image_bytes)

        image.image_model.optimized_file_path = save_filename
        return self.update_image(image)

    def delete_image(self, image: Image) -> bool:
        """Delete image from database."""
        if not image.image_model or not image.image_model.id:
            return False
        return self.database.delete_image(image.image_model.id)

    def get_processing_summary(self) -> dict:
        """Get processing status counts."""
        models = self.database.get_all_images()
        return {
            'total': len(models),
            'pending': sum(1 for m in models if m.status == ProcessingStatus.PENDING),
            'processing': sum(1 for m in models if m.status == ProcessingStatus.PROCESSING),
            'completed': sum(1 for m in models if m.status == ProcessingStatus.COMPLETED),
            'failed': sum(1 for m in models if m.status == ProcessingStatus.FAILED),
            'skipped': sum(1 for m in models if m.status == ProcessingStatus.SKIPPED)
        }

    def close(self):
        """Clean up resources."""
        self.database.close()
        self.input_storage.close()
        self.output_storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()