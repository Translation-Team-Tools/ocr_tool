import os
import sys
from pathlib import Path
from typing import Dict

from google.cloud import vision


class VisionAPIClient:
    """Handles Google Vision API interactions."""

    def __init__(self, credentials_path: str = None):
        try:
            if credentials_path:
                # Use specific credentials file
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.client = vision.ImageAnnotatorClient()
        except Exception as e:
            print(f"Failed to initialize Google Vision API client: {e}")
            print("Make sure you have set up authentication:")
            print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or")
            print("2. Run 'gcloud auth application-default login', or")
            print("3. Pass credentials_path parameter")
            sys.exit(1)

    def process_image(self, image_path: Path) -> Dict:
        """Process image with Google Vision API and return full response."""
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        # Convert response to dict for JSON serialization
        return self._response_to_dict(response)

    def _response_to_dict(self, response) -> Dict:
        """Convert Vision API response to dictionary."""
        result = {
            'full_text_annotation': None,
            'text_annotations': []
        }

        if response.full_text_annotation:
            result['full_text_annotation'] = {
                'text': response.full_text_annotation.text,
                'pages': []
            }

            for page in response.full_text_annotation.pages:
                page_data = {'blocks': []}
                for block in page.blocks:
                    block_data = {
                        'bounding_box': self._bounding_box_to_dict(block.bounding_box),
                        'paragraphs': []
                    }
                    for paragraph in block.paragraphs:
                        para_data = {
                            'bounding_box': self._bounding_box_to_dict(paragraph.bounding_box),
                            'words': []
                        }
                        for word in paragraph.words:
                            word_data = {
                                'bounding_box': self._bounding_box_to_dict(word.bounding_box),
                                'symbols': []
                            }
                            for symbol in word.symbols:
                                symbol_data = {
                                    'text': symbol.text,
                                    'confidence': symbol.confidence,
                                    'bounding_box': self._bounding_box_to_dict(symbol.bounding_box)
                                }
                                word_data['symbols'].append(symbol_data)
                            para_data['words'].append(word_data)
                        block_data['paragraphs'].append(para_data)
                    page_data['blocks'].append(block_data)
                result['full_text_annotation']['pages'].append(page_data)

        return result

    def _bounding_box_to_dict(self, bounding_box) -> Dict:
        """Convert bounding box to dictionary."""
        return {
            'vertices': [{'x': v.x, 'y': v.y} for v in bounding_box.vertices]
        }
