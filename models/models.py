from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class ProcessingStatus(Enum):
    """Status of image processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Image:
    """Represents an image in the OCR processing system."""
    file_path: str
    optimized_file_path: str
    filename: str
    file_hash: str
    vision_json_path: str = ""
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    id: Optional[int] = None