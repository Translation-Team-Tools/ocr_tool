import json
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google.cloud import vision
from google.protobuf.json_format import ParseDict
from models.models import Image, ProcessingStatus


class TextAnalyzer:
    """Analyzes OCR results for furigana detection and confidence marking."""

    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.90,
            'medium': 0.80,
            'low': 0.60
        }
        self.confidence_markers = {
            'medium': '[?]',
            'low': '[??]',
            'very_low': '[???]'
        }

    def analyze_images(self, images: List[Image]) -> List[Dict]:
        """Analyze text from all images and return structured results."""
        results = []

        for image in images:
            try:
                # Skip if no vision JSON path or failed status
                if not image.vision_json_path or image.status == ProcessingStatus.FAILED:
                    print(f"    Skipping {image.filename}: No OCR data available")
                    continue

                # Load and analyze vision response
                vision_response = self._load_vision_response(image.vision_json_path)
                if not vision_response:
                    print(f"    Skipping {image.filename}: Failed to load JSON")
                    continue

                analyzed = self._analyze_vision_response(vision_response)

                results.append({
                    'filename': image.filename,
                    'regular_paragraphs': analyzed['regular_paragraphs'],
                    'furigana_paragraphs': analyzed['furigana_paragraphs']
                })

                print(f"    Analyzed: {image.filename}")

            except Exception as e:
                print(f"    Skipping {image.filename}: Analysis error - {e}")
                continue

        return results

    def _load_vision_response(self, json_path: str) -> Optional[vision.AnnotateImageResponse]:
        """Load Vision API response from JSON file."""
        try:
            json_file = Path(json_path)
            if not json_file.exists():
                print(f"      JSON file not found: {json_path}")
                return None

            with open(json_file, 'r', encoding='utf-8') as f:
                response_dict = json.load(f)

            # Convert dict back to Vision protobuf object
            response_pb = ParseDict(response_dict, vision.AnnotateImageResponse()._pb)
            return vision.AnnotateImageResponse(response_pb)

        except Exception as e:
            print(f"      Failed to load JSON {json_path}: {e}")
            return None

    def _analyze_vision_response(self, vision_response: vision.AnnotateImageResponse) -> Dict:
        """Analyze Vision API response for text extraction and furigana detection."""
        full_text_annotation = vision_response.full_text_annotation

        if not full_text_annotation or not full_text_annotation.pages:
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        # Extract all paragraphs from Vision response
        paragraphs = []
        for page in full_text_annotation.pages:
            for block in page.blocks:
                paragraphs.extend(block.paragraphs)

        if not paragraphs:
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        # Detect furigana using refined criteria
        regular_paragraphs, furigana_paragraphs = self._detect_furigana_refined(paragraphs)

        # Process text with confidence markers
        processed_regular = [self._process_paragraph_text(p) for p in regular_paragraphs]
        processed_furigana = [self._process_paragraph_text(p) for p in furigana_paragraphs]

        # Filter out empty text
        processed_regular = [text for text in processed_regular if text.strip()]
        processed_furigana = [text for text in processed_furigana if text.strip()]

        return {
            'regular_paragraphs': processed_regular,
            'furigana_paragraphs': processed_furigana
        }

    def _detect_furigana_refined(self, paragraphs: List) -> Tuple[List, List]:
        """Refined furigana detection with stricter criteria."""
        if not paragraphs:
            return [], []

        # Collect bounding box data for analysis
        all_boxes = []
        for paragraph in paragraphs:
            if paragraph.bounding_box:
                width = self._calculate_paragraph_width(paragraph.bounding_box)
                height = self._calculate_paragraph_height(paragraph.bounding_box)
                text_content = self._extract_plain_text(paragraph)

                all_boxes.append({
                    'paragraph': paragraph,
                    'width': width,
                    'height': height,
                    'text': text_content
                })

        if not all_boxes:
            return paragraphs, []

        # Calculate size thresholds
        widths = [box['width'] for box in all_boxes if box['width'] > 0]
        heights = [box['height'] for box in all_boxes if box['height'] > 0]

        if not widths or not heights:
            return paragraphs, []

        median_width = statistics.median(widths)
        median_height = statistics.median(heights)
        width_threshold = median_width * 0.5
        height_threshold = median_height * 0.6

        regular_paragraphs = []
        furigana_paragraphs = []

        for box in all_boxes:
            paragraph = box['paragraph']
            text = box['text']

            # Strict furigana criteria - ALL must be met
            is_furigana = (
                    len(text) <= 5 and  # Very short text
                    self._is_furigana_like_text(text) and  # Mostly hiragana, no kanji
                    (box['width'] <= width_threshold or box['height'] <= height_threshold)  # Small size
            )

            if is_furigana and self._additional_furigana_score(box) >= 3:
                furigana_paragraphs.append(paragraph)
            else:
                regular_paragraphs.append(paragraph)

        return regular_paragraphs, furigana_paragraphs

    def _additional_furigana_score(self, box: Dict) -> int:
        """Calculate additional score for furigana classification."""
        score = 0
        text = box['text']

        if len(text) <= 3:
            score += 2
        if self._is_all_hiragana(text):
            score += 1
        if box['width'] > 0 and box['height'] > 0:
            ratio = box['width'] / box['height']
            if ratio > 1.5:  # Wide ratio typical for furigana
                score += 1

        return score

    def _is_furigana_like_text(self, text: str) -> bool:
        """Check if text looks like furigana (mostly hiragana, no kanji)."""
        if not text:
            return False

        hiragana_count = sum(1 for char in text if self._is_hiragana(char))
        kanji_count = sum(1 for char in text if self._is_kanji(char))
        japanese_count = sum(1 for char in text if self._is_japanese_char(char))

        if japanese_count == 0:
            return False

        # Must be at least 80% hiragana and no kanji
        return (hiragana_count / japanese_count) >= 0.8 and kanji_count == 0

    def _is_all_hiragana(self, text: str) -> bool:
        """Check if text is all hiragana characters."""
        return all(self._is_hiragana(char) for char in text if self._is_japanese_char(char))

    def _is_hiragana(self, char: str) -> bool:
        """Check if character is hiragana."""
        return '\u3040' <= char <= '\u309F'

    def _is_kanji(self, char: str) -> bool:
        """Check if character is kanji."""
        return '\u4E00' <= char <= '\u9FAF'

    def _is_japanese_char(self, char: str) -> bool:
        """Check if character is Japanese."""
        return (self._is_hiragana(char) or
                '\u30A0' <= char <= '\u30FF' or  # Katakana
                self._is_kanji(char))

    def _calculate_paragraph_width(self, bounding_box) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 2:
            return 0
        try:
            return abs(vertices[1].x - vertices[0].x)
        except (IndexError, AttributeError):
            return 0

    def _calculate_paragraph_height(self, bounding_box) -> float:
        """Calculate height of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 4:
            return 0
        try:
            return abs(vertices[2].y - vertices[0].y)
        except (IndexError, AttributeError):
            return 0

    def _extract_plain_text(self, paragraph) -> str:
        """Extract plain text from paragraph."""
        text_parts = []
        for word in paragraph.words:
            for symbol in word.symbols:
                text_parts.append(symbol.text)
        return ''.join(text_parts)

    def _process_paragraph_text(self, paragraph) -> str:
        """Extract text from paragraph with confidence markers."""
        text_parts = []

        for word in paragraph.words:
            for symbol in word.symbols:
                text = symbol.text
                confidence = getattr(symbol, 'confidence', 1.0)

                try:
                    confidence = float(confidence) if confidence is not None else 1.0
                except (ValueError, TypeError):
                    confidence = 1.0

                # Apply confidence markers
                confidence_level = self._get_confidence_level(confidence)
                if confidence_level in self.confidence_markers:
                    marker = self.confidence_markers[confidence_level]
                    text = f"{marker}{text}{marker}"

                text_parts.append(text)

        return ''.join(text_parts)

    def _get_confidence_level(self, confidence: float) -> str:
        """Classify confidence level."""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'