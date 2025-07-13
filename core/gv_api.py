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
        print(f"  → Initializing Google Vision API client...")
        print(f"    Credentials path: {self.credentials_path}")

        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            self.client = vision.ImageAnnotatorClient()
            print(f"  ✓ Google Vision API client initialized successfully")
        except Exception as e:
            print(f"  ✗ Failed to initialize Google Vision API client: {e}")
            raise

    def process_images(self, images: List[Image]) -> List[Image]:
        """
        Process images with Google Vision API.

        Takes Image objects with loaded image_bytes and populates vision_response.
        Does NOT modify status or handle storage - that's the workflow manager's job.
        """
        processed_images = []
        total_images = len(images)
        processed_count = 0
        successful_count = 0
        failed_count = 0

        print(f"  → Starting Vision API processing for {total_images} images...")

        for i, image in enumerate(images, 1):
            try:
                # Skip if no image bytes loaded or already failed
                if not image.image_bytes or image.status == ProcessingStatus.FAILED:
                    print(f"  → Skipping image {i}/{total_images}: {image.filename} (no bytes or failed)")
                    processed_images.append(image)
                    continue

                print(f"  → Processing image {i}/{total_images}: {image.filename}")
                print(f"    Image size: {len(image.image_bytes)} bytes")

                # Process with Vision API
                image.vision_response = self._call_vision_api(image.image_bytes)

                processed_images.append(image)
                processed_count += 1
                successful_count += 1
                print(f"  ✓ OCR completed: {image.filename}")

            except Exception as e:
                # Don't modify status - just set vision_response to None and let workflow manager handle it
                image.vision_response = None
                processed_images.append(image)
                processed_count += 1
                failed_count += 1
                print(f"  ✗ OCR failed for {image.filename}: {e}")

        print(f"  → Vision API processing summary:")
        print(f"    Total processed: {processed_count}/{total_images}")
        print(f"    Successful: {successful_count}")
        print(f"    Failed: {failed_count}")

        return processed_images

    def _call_vision_api(self, image_content: bytes) -> vision.AnnotateImageResponse:
        """Call Google Vision API for single image."""
        import time

        try:
            print(f"    → Sending request to Google Vision API...")
            start_time = time.time()

            image = vision.Image(content=image_content)
            response = self.client.text_detection(image=image)

            duration = time.time() - start_time
            print(f"    → API call completed in {duration:.1f}s")

            # Check for API errors
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")

            # Check if any text was detected
            if response.text_annotations:
                text_count = len(response.text_annotations)
                print(f"    → Detected {text_count} text annotations")
            else:
                print(f"    → No text detected in image")

            return response

        except Exception as e:
            print(f"    ✗ Vision API call failed: {e}")
            raise

    @staticmethod
    def get_dict(vision_response: vision.AnnotateImageResponse) -> dict:
        """Convert Vision API response to dictionary for JSON serialization."""
        return MessageToDict(vision_response._pb)