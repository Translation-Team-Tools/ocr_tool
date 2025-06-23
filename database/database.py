from pathlib import Path
from datetime import datetime
from typing import Optional, List
from peewee import *
from models.models import Image, ProcessingStatus

# Database connection
db = SqliteDatabase('result/processing.db')


class ImageRecord(Model):
    """Peewee model for image records in database."""
    file_path = CharField(unique=True)
    optimized_file_path = CharField(null=True)
    filename = CharField()
    file_hash = CharField()
    vision_json_path = CharField(default='')
    status = CharField(default='pending')
    processed_at = DateTimeField(null=True)
    error_message = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        table_name = 'images'


class DatabaseManager:
    """Database manager using Peewee ORM."""

    def __init__(self, db_path: str = "result/processing.db"):
        # Update database path if different from default
        if db_path != "result/processing.db":
            db.init(db_path)

        self._ensure_db_directory()
        self._initialize_database()

    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        db_path = Path(db.database)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        db.connect()
        db.create_tables([ImageRecord], safe=True)

    def get_image_by_path(self, file_path: str) -> Optional[Image]:
        """Get image record by file path."""
        try:
            record = ImageRecord.get(ImageRecord.file_path == file_path)
            return self._record_to_image(record)
        except ImageRecord.DoesNotExist:
            return None

    def save_image(self, image: Image) -> int:
        """Save or update image record. Returns the image ID."""
        data = {
            'file_path': image.file_path,
            'optimized_file_path': image.optimized_file_path,
            'filename': image.filename,
            'file_hash': image.file_hash,
            'vision_json_path': image.vision_json_path,
            'status': image.status.value,
            'processed_at': image.processed_at,
            'error_message': image.error_message
        }

        if image.id:
            # Update existing record
            query = ImageRecord.update(**data).where(ImageRecord.id == image.id)
            query.execute()
            return image.id
        else:
            # Create new record
            record = ImageRecord.create(**data)
            return record.id

    def get_all_images(self, status: Optional[ProcessingStatus] = None) -> List[Image]:
        """Get all images, optionally filtered by status."""
        query = ImageRecord.select()

        if status:
            query = query.where(ImageRecord.status == status.value)

        return [self._record_to_image(record) for record in query]

    def get_failed_images(self) -> List[Image]:
        """Get list of failed images."""
        query = ImageRecord.select().where(ImageRecord.status == ProcessingStatus.FAILED.value)
        return [self._record_to_image(record) for record in query]

    def _record_to_image(self, record) -> Image:
        """Convert Peewee record to Image dataclass."""
        try:
            status = ProcessingStatus(record.status)
        except ValueError:
            status = ProcessingStatus.PENDING

        return Image(
            id=record.id,
            file_path=record.file_path,
            optimized_file_path=record.optimized_file_path or '',
            filename=record.filename,
            file_hash=record.file_hash,
            vision_json_path=record.vision_json_path,
            status=status,
            processed_at=record.processed_at,
            error_message=record.error_message
        )

    def close(self):
        """Close database connection."""
        db.close()