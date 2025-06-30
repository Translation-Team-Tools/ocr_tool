import json
from pathlib import Path
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
        """Process images with Google Vision API and save responses as JSON."""
        processed_images = []

        for image in images:
            try:
                # Skip if no optimized path (failed optimization)
                if not image.optimized_file_path or image.status == ProcessingStatus.FAILED:
                    processed_images.append(image)
                    continue

                # Process with Vision API
                image.vision_response = self._call_vision_api(image.optimized_file_path)

                # Update image model
                image.status = ProcessingStatus.COMPLETED

                processed_images.append(image)
                print(f"    OCR completed: {image.filename}")

            except Exception as e:
                # Skip failed image and continue
                image.status = ProcessingStatus.FAILED
                image.error_message = f"Vision API error: {str(e)}"
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
        return MessageToDict(vision_response._pb)

# # Save as JSON
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(response_dict, f, ensure_ascii=False, indent=2)
#
# return json_path