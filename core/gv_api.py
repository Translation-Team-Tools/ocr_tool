from typing import List

from google.cloud import vision
from google.protobuf.json_format import MessageToDict

from data.models import Image, ProcessingStatus
from utils.logger import logger


class VisionProcessor:
    """Handles Google Vision API OCR processing."""

    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Vision API client with timeout settings."""
        import os
        from google.api_core import client_options

        logger.info("Initializing Google Vision API client")

        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path

            # Set client options with timeout
            client_opts = client_options.ClientOptions(
                api_endpoint="vision.googleapis.com"
            )

            self.client = vision.ImageAnnotatorClient(client_options=client_opts)
            logger.success("Google Vision API client ready")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision API client: {e}")
            raise

    def process_images(self, images: List[Image]) -> List[Image]:
        """
        Process images with Google Vision API.

        Takes Image objects with loaded image_bytes and populates vision_response.
        Does NOT modify status or handle storage - that's the workflow manager's job.
        """
        processed_images = []
        total_images = len(images)
        successful_count = 0
        failed_count = 0

        logger.info(f"Processing {total_images} images with Vision API")

        for i, image in enumerate(images, 1):
            try:
                # Skip if no image bytes loaded or already failed
                if not image.image_bytes or image.status == ProcessingStatus.FAILED:
                    processed_images.append(image)
                    continue

                logger.progress(i, total_images, image.filename)

                # Process with Vision API
                image.vision_response = self._call_vision_api(image.image_bytes)
                processed_images.append(image)
                successful_count += 1

            except Exception as e:
                # Don't modify status - just set vision_response to None and let workflow manager handle it
                image.vision_response = None
                processed_images.append(image)
                failed_count += 1
                logger.error(f"OCR failed for {image.filename}: {e}")

        if failed_count > 0:
            logger.warning(f"Vision API completed: {successful_count} successful, {failed_count} failed")
        else:
            logger.success(f"Vision API completed: {successful_count} images processed")

        return processed_images

    def _call_vision_api(self, image_content: bytes) -> vision.AnnotateImageResponse:
        """Call Google Vision API for single image with no timeout restrictions."""
        import time

        start_time = time.time()

        # Create image object
        image = vision.Image(content=image_content)

        try:
            # Make the API call with NO timeout or retry restrictions
            response = self.client.document_text_detection(
                image=image)  # IT'S IMPORTANT TO USE document_text_detection()

            # Check for API errors
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Vision API call failed after {duration:.1f}s: {e}")
            raise

    @staticmethod
    def get_dict(vision_response: vision.AnnotateImageResponse) -> dict:
        """Convert Vision API response to dictionary for JSON serialization."""
        return MessageToDict(
            vision_response._pb,
            always_print_fields_with_no_presence=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=False
        )