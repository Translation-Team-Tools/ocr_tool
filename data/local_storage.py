import hashlib
from typing import List

from fs import open_fs
from fs.path import join, splitext, dirname

supported_formats = {'.jpg', '.jpeg', '.png'}


class LocalStorage:

    def __init__(self, base_path: str):
        """
        Initialize LocalStorage with a base path.

        Args:
            base_path: Base directory for all file operations
        """
        self.base_path = base_path
        self.fs = open_fs(base_path, create=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the filesystem connection."""
        if hasattr(self, 'fs') and self.fs:
            self.fs.close()

    def discover_files_recursive(self, subfolder: str = '/') -> List[str]:
        """
        Discover supported files recursively in folder with depth-first traversal.

        Args:
            subfolder: Subfolder within base path to scan (default: root)

        Returns:
            List of relative file paths in filesystem order
        """
        files = []

        def scan_directory(current_path: str):
            try:
                # Get sorted items
                items = sorted(self.fs.listdir(current_path))

                # Process files first
                for item in items:
                    item_path = join(current_path, item)
                    if self.fs.isfile(item_path):
                        _, ext = splitext(item)
                        if ext.lower() in supported_formats:
                            files.append(item_path)

                # Then process subdirectories
                for item in items:
                    item_path = join(current_path, item)
                    if self.fs.isdir(item_path):
                        scan_directory(item_path)

            except Exception as e:
                print(f"Warning: Cannot access {current_path}: {e}")

        scan_directory(subfolder)
        return files

    def ensure_directory_exists(self, path: str) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            path: Directory path relative to base path
        """
        dir_path = dirname(path)
        if dir_path and not self.fs.exists(dir_path):
            self.fs.makedirs(dir_path, recreate=True)

    def get_file_size(self, path: str) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path relative to base path

        Returns:
            File size in bytes
        """
        return self.fs.getsize(path)

    def file_exists(self, path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: File path relative to base path

        Returns:
            True if file exists
        """
        return self.fs.exists(path)

    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file.

        Args:
            file_path: Path to file relative to base path

        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()

        with self.fs.open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def read_file_bytes(self, path: str) -> bytes:
        """
        Read file as bytes.

        Args:
            path: File path relative to base path

        Returns:
            File contents as bytes
        """
        with self.fs.open(path, 'rb') as f:
            return f.read()

    def write_file_bytes(self, path: str, data: bytes) -> None:
        """
        Write bytes to file.

        Args:
            path: File path relative to base path
            data: Bytes to write
        """
        self.ensure_directory_exists(path)
        with self.fs.open(path, 'wb') as f:
            f.write(data)

    def copy_file(self, source_path: str, destination: str) -> None:
        """
        Copy file within the storage.

        Args:
            source_path: Source file path relative to base path
            destination: Destination file path relative to base path
        """
        self.ensure_directory_exists(destination)
        self.fs.copy(source_path, destination)

    def delete_file(self, path: str) -> None:
        """
        Delete file from storage.

        Args:
            path: File path relative to base path
        """
        if self.fs.exists(path):
            self.fs.remove(path)

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Get absolute system path for a relative path.

        Args:
            relative_path: Path relative to base path

        Returns:
            Absolute system path
        """
        return self.fs.getsyspath(relative_path)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string (e.g., "1.5MB")
        """
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.1f}MB"

    @staticmethod
    def calculate_compression_ratio(original_size: int, optimized_size: int) -> float:
        """
        Calculate compression ratio as percentage.

        Args:
            original_size: Original file size in bytes
            optimized_size: Optimized file size in bytes

        Returns:
            Compression ratio as percentage (0-100)
        """
        if original_size == 0:
            return 0
        return (1 - optimized_size / original_size) * 100