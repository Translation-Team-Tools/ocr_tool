from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker

from data.models import Base, Image, ProcessingStatus


class Database:
    def __init__(self, folder_path: str, project_root: str = None):
        """
        Initialize database for the given folder path.
        Creates SQLite file in result/{subfolder}/ directory.

        Args:
            folder_path: Path to the folder being processed
            project_root: Path to project root (optional, defaults to current working directory)
        """
        self.folder_path = Path(folder_path)
        self.subfolder = self.folder_path.name

        # Use provided project root or current working directory
        if project_root:
            project_root_path = Path(project_root)
        else:
            project_root_path = Path.cwd()

        # Create result directory structure in project root
        self.result_dir = project_root_path / "result" / self.subfolder
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database path
        self.db_path = self.result_dir / "images.db"
        self.db_url = f"sqlite:///{self.db_path}"

        # Create engine and session factory
        self.engine = create_engine(
            self.db_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        self.create_tables()

    def create_tables(self):
        """Create database tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_image(self, image: Image) -> Image:
        """
        Add an existing Image object to database.

        Args:
            image: Image object to add

        Returns:
            Added Image object with ID assigned
        """
        with self.get_session() as session:
            session.add(image)
            session.flush()
            session.refresh(image)
            session.expunge(image)
            return image

    def get_image_by_id(self, image_id: int) -> Optional[Image]:
        """Get image by ID."""
        with self.get_session() as session:
            img = session.query(Image).filter(Image.id == image_id).first()
            if img:
                session.expunge(img)
            return img

    def get_image_by_hash(self, file_hash: str) -> Optional[Image]:
        """Get image by file hash."""
        with self.get_session() as session:
            img = session.query(Image).filter(Image.file_hash == file_hash).first()
            if img:
                session.expunge(img)
            return img

    def get_image_by_filename(self, filename: str) -> Optional[Image]:
        """Get image by filename."""
        with self.get_session() as session:
            img = session.query(Image).filter(Image.filename == filename).first()
            if img:
                session.expunge(img)
            return img

    def get_images_by_status(self, status: ProcessingStatus) -> List[Image]:
        """Get images by processing status."""
        with self.get_session() as session:
            images = session.query(Image).filter(Image.status == status).all()
            for img in images:
                session.expunge(img)
            return images

    def get_image(self, image_id: int = None, file_hash: str = None, filename: str = None, status: ProcessingStatus = None):
        if image_id: return self.get_image_by_id(image_id=image_id)
        if file_hash: return self.get_image_by_hash(file_hash=file_hash)
        if filename: return self.get_image_by_filename(filename=filename)
        if status: return self.get_images_by_status(status=status)
        return False

    def get_all_images(self) -> List[Image]:
        """Get all images."""
        with self.get_session() as session:
            images = session.query(Image).all()
            for img in images:
                session.expunge(img)
            return images

    def update_image(self, image: Image) -> Image:
        """
        Update existing image in database.

        Args:
            image: Image object with updated data

        Returns:
            Updated Image object
        """
        with self.get_session() as session:
            merged_image = session.merge(image)  # Assign the merged instance
            session.flush()
            session.refresh(merged_image)  # Refresh to get updated data
            session.expunge(merged_image)  # Expunge the merged instance
            return merged_image  # Return the merged instance

    def delete_image(self, image_id: int) -> bool:
        """
        Delete image by ID.

        Args:
            image_id: ID of image to delete

        Returns:
            True if deleted, False if not found
        """
        with self.get_session() as session:
            img = session.query(Image).filter(Image.id == image_id).first()
            if not img:
                return False
            session.delete(img)
            return True

    def search_images(self, search_term: str) -> List[Image]:
        """
        Search images by filename or path.

        Args:
            search_term: Term to search for

        Returns:
            List of matching Image objects
        """
        with self.get_session() as session:
            images = session.query(Image).filter(
                or_(
                    Image.filename.like(f"%{search_term}%"),
                    Image.original_file_path.like(f"%{search_term}%")
                )
            ).all()
            for img in images:
                session.expunge(img)
            return images

    def close(self):
        """Close database connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose()


# Example usage:
if __name__ == "__main__":
    # Initialize database
    db = Database("/path/to/your/image/folder")

    # Create Image object yourself
    image = Image(
        original_file_path="/path/to/original.jpg",
        optimized_file_path="/path/to/optimized.jpg",
        filename="image.jpg",
        file_hash="your_hash_here",
        vision_json_path="/path/to/vision.json",
        status=ProcessingStatus.PENDING
    )

    # Add to database
    saved_image = db.add_image(image)

    # Retrieve from database
    retrieved_image = db.get_image(image_id=saved_image.id)

    # Update image
    retrieved_image.status = ProcessingStatus.COMPLETED
    db.update_image(retrieved_image)

    # Get all pending images
    pending_images = db.get_image(status=ProcessingStatus.PENDING)

    # Delete image
    db.delete_image(saved_image.id)

    # Clean up
    db.close()