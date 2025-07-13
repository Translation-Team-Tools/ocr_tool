from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from google.cloud.vision import AnnotateImageResponse

Base = declarative_base()


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ImageModel(Base):
    """Database model for image metadata"""
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_file_path = Column(String(512), nullable=False)
    optimized_file_path = Column(String(512), nullable=True, default="")
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(128), nullable=False)
    vision_json_path = Column(String(512), nullable=True, default="")
    processed_at = Column(DateTime, nullable=True)
    status = Column(SQLEnum(ProcessingStatus), nullable=False, default=ProcessingStatus.PENDING)

    def __repr__(self):
        return f"<ImageModel(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"


@dataclass
class Image:
    """
    Main image data container that combines file data with database metadata.
    Used by StorageManager as the primary interface for image operations.
    """
    image_bytes: Optional[bytes] = None
    vision_response: Optional[AnnotateImageResponse] = None
    analysis_results: Optional[List[str]] = None
    image_model: Optional[ImageModel] = None

    @property
    def filename(self) -> str:
        """Get filename from image model."""
        if not self.image_model:
            raise ValueError("Image must have image_model to access filename")
        return self.image_model.filename

    @property
    def file_hash(self) -> str:
        """Get file hash from image model."""
        if not self.image_model:
            raise ValueError("Image must have image_model to access file_hash")
        return self.image_model.file_hash

    @property
    def status(self) -> ProcessingStatus:
        """Get processing status from image model."""
        if not self.image_model:
            raise ValueError("Image must have image_model to access status")
        return self.image_model.status

    @property
    def original_file_path(self) -> str:
        """Get original file path from image model."""
        if not self.image_model:
            raise ValueError("Image must have image_model to access original_file_path")
        return self.image_model.original_file_path

    @property
    def optimized_file_path(self) -> str:
        """Get optimized file path from image model."""
        if not self.image_model:
            raise ValueError("Image must have image_model to access optimized_file_path")
        return self.image_model.optimized_file_path

    @property
    def id(self) -> Optional[int]:
        """Get database ID from image model."""
        if not self.image_model:
            return None
        return self.image_model.id

    def has_image_data(self) -> bool:
        """Check if image has actual image bytes loaded."""
        return self.image_bytes is not None

    def has_vision_data(self) -> bool:
        """Check if image has vision analysis results."""
        return self.vision_response is not None

    def has_analysis_results(self) -> bool:
        """Check if image has analysis results."""
        return self.analysis_results is not None and len(self.analysis_results) > 0

    def is_optimized(self) -> bool:
        """Check if image has been optimized (has optimized file path)."""
        return (self.image_model is not None and
                self.image_model.optimized_file_path is not None and
                self.image_model.optimized_file_path != "")

    def __repr__(self):
        model_info = f"id={self.id}, filename='{self.filename}'" if self.image_model else "no_model"
        data_info = f"bytes={'loaded' if self.has_image_data() else 'none'}"
        vision_info = f"vision={'yes' if self.has_vision_data() else 'no'}"
        return f"<Image({model_info}, {data_info}, {vision_info})>"