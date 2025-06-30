import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from google.cloud import vision

from config.config import Confidence
from data.models import Image


# Parameters to analyze
# - is all japanese characters
# - zero kanji
# - is kana amount > 90%
# - word length <= 6
# - width lowen than 80% (60%?) of median width

"""
ratio = paragraph_median / page_median

# Main furigana detection
if ratio > 0.8:
    return False  # Too similar in size

# Safety: difference must be meaningful relative to text size
relative_diff = (page_median - paragraph_median) / page_median
if relative_diff < 0.15:  # Less than 15% smaller
    return False  # Probably just OCR variance

# Safety: very small paragraphs
if paragraph_char_count < 7:
    return False
"""

@dataclass
class _Symbol:
    width: float
    text: str

@dataclass
class _Paragraph:
    symbols: List[_Symbol] = field(default_factory=list)
    furigana: bool = True


class TextAnalyzer:
    """Analyzes OCR results for furigana detection and confidence marking."""

    def analyze_images(self, images: List[Image]) -> None:
        """Analyze text from all images and return structured results."""
        for image in images:
            try:
                self._analyze_vision_response(image.vision_response)
                print(f"    Analyzed: {image.filename}")

            except Exception as e:
                print(f"    Skipping {image.filename}: Analysis error - {e}")
                continue

    def _analyze_vision_response(self, vision_response: vision.AnnotateImageResponse) -> Dict | None:
        """Analyze Vision API response for text extraction and furigana detection."""
        full_text_annotation = vision_response.full_text_annotation

        if not full_text_annotation or not full_text_annotation.pages:
            return None

        paragraphs: List[_Paragraph] = []

        # Extract all paragraphs from Vision response
        for page in full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    par = _Paragraph()
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            width = self._calculate_width(symbol.bounding_box)
                            par.symbols.append(_Symbol(width=width, text=symbol.text))
                            if not self._is_furigana_char(symbol.text):
                                par.furigana = False
                    paragraphs.append(par)
        return None


    def _is_hiragana(self, char: str) -> bool:
        """Check if character is hiragana."""
        return '\u3040' <= char <= '\u309F'

    def _is_katakana(self, char: str) -> bool:
        """Check if character is katakana."""
        return '\u30A0' <= char <= '\u30FF'

    def _is_kanji(self, char: str) -> bool:
        """Check if character is kanji."""
        return '\u4E00' <= char <= '\u9FAF'

    def _is_furigana_char(self, char: str) -> bool:
        """Check if character is Japanese."""
        return (self._is_hiragana(char) or self._is_katakana(char)) and not self._is_kanji(char)

    def _calculate_width(self, bounding_box) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 2:
            return 0
        try:
            return abs(vertices[1].x - vertices[0].x)
        except (IndexError, AttributeError):
            return 0

    def _calculate_height(self, bounding_box) -> float:
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
                confidence_level: Confidence = self._get_confidence_level(confidence)
                if confidence_level is not None:
                    marker = confidence_level.value.marker
                    text = f"{marker}{text}{marker}"

                text_parts.append(text)

        return ''.join(text_parts)

    def _get_confidence_level(self, confidence: float) -> Confidence.value:
        """Classify confidence level."""
        for confidence_level in Confidence:
            if confidence >= confidence_level.value.threshold:
                return confidence_level
        return None