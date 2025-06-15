import datetime
import sqlite3
from pathlib import Path
from typing import List, Dict


class DatabaseManager:
    """Manages SQLite database for tracking processed images."""

    def __init__(self, db_path: str = "result/processing.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()

    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_images (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    processed_at TIMESTAMP,
                    vision_json_path TEXT,
                    status TEXT,
                    error_message TEXT,
                    UNIQUE(file_path, file_hash)
                )
            """)
            conn.commit()

    def is_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if image was already processed successfully."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT status FROM processed_images WHERE file_path = ? AND file_hash = ?",
                (file_path, file_hash)
            )
            result = cursor.fetchone()
            return result and result[0] == 'completed'

    def add_processed_image(self, file_path: str, filename: str, file_hash: str,
                            vision_json_path: str, status: str = 'completed',
                            error_message: str = None):
        """Add or update processed image record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_images 
                (file_path, filename, file_hash, processed_at, vision_json_path, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, filename, file_hash, datetime.now().isoformat(),
                  vision_json_path, status, error_message))
            conn.commit()

    def get_failed_images(self) -> List[Dict]:
        """Get list of failed images for retry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path, filename, error_message FROM processed_images WHERE status = 'failed'"
            )
            return [{'path': row[0], 'filename': row[1], 'error': row[2]}
                    for row in cursor.fetchall()]
