from typing import List

from google.cloud import vision
from google.protobuf.json_format import MessageToDict

from data.models import Image, ProcessingStatus


class VisionProcessor:
    """Handles Google Vision API OCR processing."""

    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Vision API client."""
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        self.client = vision.ImageAnnotatorClient()

    def process_images(self, images: List[Image]) -> List[Image]:
        """
        Process images with Google Vision API.

        Takes Image objects with loaded image_bytes and populates vision_response.
        Does NOT modify status or handle storage - that's the workflow manager's job.
        """
        processed_images = []

        for image in images:
            try:
                # Skip if no image bytes loaded or already failed
                if not image.image_bytes or image.status == ProcessingStatus.FAILED:
                    processed_images.append(image)
                    continue

                # Process with Vision API
                image.vision_response = self._call_vision_api(image.image_bytes)

                processed_images.append(image)
                print(f"    OCR completed: {image.filename}")

            except Exception as e:
                # Don't modify status - just set vision_response to None and let workflow manager handle it
                image.vision_response = None
                processed_images.append(image)
                print(f"    OCR failed for {image.filename}: {e}")

        return processed_images

    def _call_vision_api(self, image_content: bytes) -> vision.AnnotateImageResponse:
        """Call Google Vision API for single image."""
        image = vision.Image(content=image_content)
        response = self.client.text_detection(image=image)

        # Check for API errors
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        return response

    @staticmethod
    def get_dict(vision_response: vision.AnnotateImageResponse) -> dict:
        """Convert Vision API response to dictionary for JSON serialization."""
        return MessageToDict(vision_response._pb)