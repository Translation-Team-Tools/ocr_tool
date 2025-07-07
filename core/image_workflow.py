# Import the StorageManager
import argparse

from data.storage_manager import StorageManager
from data.models import ProcessingStatus


class ImageWorkflowManager:

    # Basic workflow example
    def process_images_workflow(input_folder: str):
        """
        Complete workflow for discovering, optimizing, and saving images.

        Args:
            input_folder: Path to folder containing images to process
        """

        # 1. Initialize StorageManager
        with StorageManager(
                input_folder_path=input_folder,
                output_folder="optimized_images",  # Will be created in result/ folder
                project_root=None  # Uses current working directory
        ) as storage:
            # 2. Discover and save images to database
            print("🔍 Discovering images...")
            images = storage.discover_and_save_images()

            if not images:
                print("No images found!")
                return

            # 3. Show initial summary
            storage.print_processing_summary()

            # 4. Optimize all pending images
            print("\n🎨 Optimizing images...")
            optimized_images = storage.optimize_images_batch(
                max_width=2048,
                jpeg_quality=90,
                enhance_contrast=True,
                convert_to_grayscale=False
            )

            # 5. Show final summary
            storage.print_processing_summary()

            return optimized_images


    # Advanced workflow with custom processing
    def advanced_workflow(input_folder: str):
        """
        Advanced workflow with more control over the process.
        """

        storage = StorageManager(input_folder_path=input_folder)

        try:
            # Discover images
            images = storage.discover_and_save_images()

            # Process images individually with custom settings
            for image in images:
                if image.status == ProcessingStatus.PENDING:
                    print(f"Processing {image.filename}...")

                    # Custom optimization settings per image
                    if "document" in image.filename.lower():
                        # For documents: grayscale, high contrast
                        storage.optimize_image(
                            image=image,
                            max_width=2048,
                            jpeg_quality=95,
                            enhance_contrast=True,
                            convert_to_grayscale=True
                        )
                    else:
                        # For photos: preserve color, moderate quality
                        storage.optimize_image(
                            image=image,
                            max_width=1600,
                            jpeg_quality=85,
                            enhance_contrast=False,
                            convert_to_grayscale=False
                        )

            # Get processing results
            completed_images = storage.get_image(status=ProcessingStatus.COMPLETED)
            failed_images = storage.get_image(status=ProcessingStatus.FAILED)

            print(f"✅ Successfully processed: {len(completed_images)}")
            print(f"❌ Failed to process: {len(failed_images)}")

            return completed_images

        finally:
            storage.close()


    # Simple function for quick optimization
    def quick_optimize(input_folder: str):
        """
        Quick optimization with default settings.
        """
        with StorageManager(input_folder_path=input_folder) as storage:
            images = storage.discover_and_save_images()
            return storage.optimize_images_batch()


    # Working with existing database
    def resume_processing(input_folder: str):
        """
        Resume processing from where you left off.
        """
        with StorageManager(input_folder_path=input_folder) as storage:
            # Get pending images (previously discovered but not processed)
            pending_images = storage.get_image(status=ProcessingStatus.PENDING)

            if pending_images:
                print(f"Resuming processing of {len(pending_images)} pending images...")
                storage.optimize_images_batch(pending_images)
            else:
                print("No pending images to process!")


    # Search and filter images
    def search_and_process(input_folder: str, search_term: str):
        """
        Search for specific images and process them.
        """
        with StorageManager(input_folder_path=input_folder) as storage:
            # Search for images
            found_images = storage.search_images(search_term)

            if found_images:
                print(f"Found {len(found_images)} images matching '{search_term}'")

                # Process only pending ones
                pending_found = [img for img in found_images if img.status == ProcessingStatus.PENDING]
                if pending_found:
                    storage.optimize_images_batch(pending_found)
                else:
                    print("All found images are already processed!")
            else:
                print(f"No images found matching '{search_term}'")


    # Get file paths for further processing
    def get_optimized_paths(input_folder: str):
        """
        Get paths to optimized images for further OCR processing.
        """
        with StorageManager(input_folder_path=input_folder) as storage:
            completed_images = storage.get_image(status=ProcessingStatus.COMPLETED)

            optimized_paths = []
            for image in completed_images:
                optimized_path = storage.get_optimized_file_path(image)
                if optimized_path and storage.optimized_file_exists(image):
                    optimized_paths.append(optimized_path)

            return optimized_paths


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to the folder')
    args = parser.parse_args()

    input_folder = args.input_path

    # Simple workflow
    image_list = ImageWorkflowManager.process_images_workflow(input_folder)


    # Or get optimized paths for OCR
    optimized_paths = ImageWorkflowManager.get_optimized_paths(input_folder)
    print(f"Ready for OCR: {len(optimized_paths)} images")

    # Resume if needed
    ImageWorkflowManager.resume_processing(input_folder)