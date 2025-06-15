import os
import sys
from pathlib import Path
from typing import Dict, Any

from google.cloud import vision
from google.protobuf.json_format import MessageToDict


class VisionAPIClient:
    """Handles Google Vision API interactions."""

    def __init__(self, credentials_path: str = None) -> None:
        try:
            if credentials_path:
                # Convert to absolute path and check if exists
                abs_path = os.path.abspath(credentials_path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"Credentials file not found: {abs_path}")
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_path
            self.client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient()
        except Exception as e:
            print(f"Failed to initialize Google Vision API client: {e}")
            print("Make sure you have set up authentication:")
            print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or")
            print("2. Run 'gcloud auth application-default login', or")
            print("3. Pass credentials_path parameter")
            sys.exit(1)

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process image with Google Vision API and return full response."""
        import time

        print(f"  Starting API call for {image_path.name}...")
        start_time = time.time()

        # Read the already-optimized image file
        with open(image_path, 'rb') as image_file:
            content: bytes = image_file.read()

        file_size_mb = len(content) / (1024 * 1024)
        print(f"  Image size: {file_size_mb:.1f}MB")

        # Create Vision API image object
        image = vision.Image(content=content)  # type: ignore

        # API call - should be fast with pre-optimized images
        api_start = time.time()
        response = self.client.document_text_detection(image=image)  # type: ignore
        api_time = time.time() - api_start
        print(f"  ✓ API response received in {api_time:.1f}s")

        if hasattr(response, 'error') and response.error and response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        # Convert response to dict
        convert_start = time.time()
        result = MessageToDict(response._pb)  # type: ignore
        convert_time = time.time() - convert_start

        total_time = time.time() - start_time
        print(f"  Total API time: {total_time:.1f}s")

        return result