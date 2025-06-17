import json
import os
import sys
from pathlib import Path

from google.cloud import vision
from google.protobuf.json_format import MessageToDict, ParseDict, MessageToJson, Parse


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

    def process_image(self, image_path: Path) -> vision.AnnotateImageResponse:
        """Process image with Google Vision API and return Vision object."""
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

        total_time = time.time() - start_time
        print(f"  Total API time: {total_time:.1f}s")

        return response

    def save_response_as_json(self, response: vision.AnnotateImageResponse, json_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Direct protobuf to JSON string
        json_string = MessageToJson(response._pb)

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_string)

    def load_response_from_json(self, json_path: Path) -> vision.AnnotateImageResponse:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            json_string = f.read()

        # Direct JSON string to protobuf
        response = vision.AnnotateImageResponse()
        Parse(json_string, response)
        return response