from enum import Enum
from typing import List

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


class Image(Base):
    __tablename__ = 'images'
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_file_path = Column(String(512), nullable=False)
    optimized_file_path = Column(String(512), nullable=False)
    image_bytes: bytes
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(128), nullable=False)
    vision_response: AnnotateImageResponse
    vision_json_path = Column(String(512), nullable=False, default="")
    analysis_results: List[str]
    processed_at = Column(DateTime, nullable=True)
    status = Column(SQLEnum(ProcessingStatus), nullable=False, default=ProcessingStatus.PENDING)

    def __repr__(self):
        return f"<Image(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"