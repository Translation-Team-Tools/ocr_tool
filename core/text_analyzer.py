import statistics
from dataclasses import dataclass, field
from typing import List

from google.cloud import vision

from config.config import Confidence
from output_generation.output_generator import OutputGenerator
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
    furigana_chars: bool = True
    widths: List[float] = field(default_factory=list)


class TextAnalyzer:
    """Analyzes OCR results for furigana detection and confidence marking."""

    def __init__(self):
        self.output_generator = OutputGenerator()

    def analyze_images(self, images: List[Image]) -> str:
        """Analyze text from all images and return structured results."""
        image_sections: List[str] = []

        for image in images:
            try:
                paragraphs = self._analyze_full_text_annotation(image.vision_response)
                section = self._build_output(paragraphs)

                image_sections.append(self.output_generator.build_image_section(lines=section, filename=image.filename))
                print(f"    Analyzed: {image.filename}")

            except Exception as e:
                print(f"    Skipping {image.filename}: Analysis error - {e}")
                continue

        return self.output_generator.build_final_result(image_sections=image_sections)

    def _analyze_full_text_annotation(self, vision_response: vision.AnnotateImageResponse) -> List[_Paragraph] | None:
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
                        wrd = _Word()
                        for symbol in word.symbols:
                            width = AnalyzerUtils.calculate_width(symbol.bounding_box)
                            par.widths.append(width)
                            confidence = AnalyzerUtils.get_confidence_level(confidence=symbol.confidence)
                            wrd.symbols.append(_Symbol(confidence=confidence, text=symbol.text))
                            if not AnalyzerUtils.is_furigana_char(symbol.text):
                                par.furigana_chars = False
                        par.words.append(wrd)
                    paragraphs.append(par)
        return paragraphs

    def _build_output(self, paragraphs: List[_Paragraph]) -> List[str]:
        page_median = statistics.median([width for paragraph in paragraphs for width in paragraph.widths])

        lines: List[str] = []

        for paragraph in paragraphs:
            paragraph_median = statistics.median(paragraph.widths)
            ratio = paragraph_median / page_median
            furigana = ratio < 0.8 and paragraph.furigana_chars and not any(len(word.symbols) > 6 for word in paragraph.words)

            par_text = ""

            for word in paragraph.words:
                wrd_text = ""
                for char in word.symbols:
                    char.text = self.output_generator.mark_char(text=char.text, marker=char.confidence.value.marker)
                    wrd_text += char.text

                if furigana:
                    wrd_text = f' {wrd_text} '

                par_text += wrd_text

            par_text = self.output_generator.build_line(text=par_text, is_furigana=furigana)

            lines.append(par_text)
        return lines



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