import statistics
from dataclasses import dataclass, field
from typing import List, Dict

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
    confidence: Confidence
    text: str

@dataclass
class _Word:
    symbols: List[_Symbol] = field(default_factory=list)

@dataclass
class _Paragraph:
    words: List[_Word] = field(default_factory=list)
    furigana: bool = True
    widths: List[float] = field(default_factory=list)


class TextAnalyzer:
    """Analyzes OCR results for furigana detection and confidence marking."""

    def __init__(self):
        self.paragraphs: List[_Paragraph] = []

    def analyze_images(self, images: List[Image]) -> None:
        """Analyze text from all images and return structured results."""
        for image in images:
            try:
                self._analyze_full_text_annotation(image.vision_response)
                print(f"    Analyzed: {image.filename}")

            except Exception as e:
                print(f"    Skipping {image.filename}: Analysis error - {e}")
                continue

    def _analyze_full_text_annotation(self, vision_response: vision.AnnotateImageResponse) -> Dict | None:
        """Analyze Vision API response for text extraction and furigana detection."""
        full_text_annotation = vision_response.full_text_annotation

        if not full_text_annotation or not full_text_annotation.pages:
            return None

        # Extract all paragraphs from Vision response
        for page in full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    par = _Paragraph()
                    for word in paragraph.words:
                        wrd = _Word()
                        for symbol in word.symbols:
                            width = AnalyzerUtils.calculate_width(symbol.bounding_box)
                            par.widths.append(width)
                            confidence = AnalyzerUtils.get_confidence_level(confidence=symbol.confidence)
                            wrd.symbols.append(_Symbol(confidence=confidence, text=symbol.text))
                            if not AnalyzerUtils.is_furigana_char(symbol.text):
                                par.furigana = False
                        par.words.append(wrd)
                    self.paragraphs.append(par)
        return None

class AnalyzerUtils:
    @staticmethod
    def _is_hiragana(char: str) -> bool:
        """Check if character is hiragana."""
        return '\u3040' <= char <= '\u309F'

    @staticmethod
    def _is_katakana(char: str) -> bool:
        """Check if character is katakana."""
        return '\u30A0' <= char <= '\u30FF'

    @staticmethod
    def _is_kanji(char: str) -> bool:
        """Check if character is kanji."""
        return '\u4E00' <= char <= '\u9FAF'

    @staticmethod
    def is_furigana_char(char: str) -> bool:
        """Check if character is Japanese."""
        return (AnalyzerUtils._is_hiragana(char) or AnalyzerUtils._is_katakana(char)) and not AnalyzerUtils._is_kanji(char)

    @staticmethod
    def calculate_width(bounding_box) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 2:
            return 0
        try:
            return abs(vertices[1].x - vertices[0].x)
        except (IndexError, AttributeError):
            return 0

    @staticmethod
    def calculate_height(bounding_box) -> float:
        """Calculate height of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 4:
            return 0
        try:
            return abs(vertices[2].y - vertices[0].y)
        except (IndexError, AttributeError):
            return 0

    @staticmethod
    def get_confidence_level(confidence: float) -> Confidence.value:
        """Classify confidence level."""
        for confidence_level in Confidence:
            if confidence >= confidence_level.value.threshold:
                return confidence_level
        return None