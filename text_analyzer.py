import statistics
from typing import Dict, Tuple, List

from google.cloud import vision


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

    def analyze_vision_response(self, vision_response: vision.AnnotateImageResponse) -> Dict:
        """Analyze Vision API response for text extraction and furigana detection."""
        # Work with proper Vision object
        full_text_annotation = vision_response.full_text_annotation

        if not full_text_annotation:
            print("  Warning: No text annotation found in response")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        # Extract all paragraphs from Vision object
        paragraphs = []
        pages = full_text_annotation.pages

        if not pages:
            print("  Warning: No pages found in text annotation")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        for page in pages:
            blocks = page.blocks
            for block in blocks:
                paragraphs.extend(block.paragraphs)

        if not paragraphs:
            print("  Warning: No paragraphs found in pages")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        print(f"  Found {len(paragraphs)} paragraphs to analyze")

        # Use refined furigana detection with stricter criteria
        regular_paragraphs, furigana_paragraphs = self._detect_furigana_refined(paragraphs)

        # Process text with confidence markers
        processed_regular = []
        processed_furigana = []

        for p in regular_paragraphs:
            text = self._process_paragraph_text(p)
            if text.strip():  # Only add non-empty text
                processed_regular.append(text)

        for p in furigana_paragraphs:
            text = self._process_paragraph_text(p)
            if text.strip():  # Only add non-empty text
                processed_furigana.append(text)

        print(f"  Processed: {len(processed_regular)} regular lines, {len(processed_furigana)} furigana lines")

        return {
            'regular_paragraphs': processed_regular,
            'furigana_paragraphs': processed_furigana
        }

    def _detect_furigana_refined(self, paragraphs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Refined furigana detection with stricter criteria to avoid false positives."""
        if not paragraphs:
            return [], []

        # Collect all bounding box data for analysis
        all_boxes = []
        for paragraph in paragraphs:
            bounding_box = paragraph.bounding_box
            if bounding_box:
                width = self._calculate_paragraph_width(bounding_box)
                height = self._calculate_paragraph_height(bounding_box)
                y_position = self._get_paragraph_y_position(bounding_box)

                all_boxes.append({
                    'paragraph': paragraph,
                    'width': width,
                    'height': height,
                    'y_position': y_position,
                    'bounding_box': bounding_box
                })

        if not all_boxes:
            return paragraphs, []

        # Calculate statistics for classification
        widths = [box['width'] for box in all_boxes if box['width'] > 0]
        heights = [box['height'] for box in all_boxes if box['height'] > 0]

        if not widths or not heights:
            return paragraphs, []

        median_width = statistics.median(widths)
        median_height = statistics.median(heights)

        # More conservative thresholds
        width_threshold = median_width * 0.5  # Much stricter width requirement
        height_threshold = median_height * 0.6  # Stricter height requirement

        regular_paragraphs = []
        furigana_paragraphs = []

        print(f"  Detection thresholds: width <= {width_threshold:.1f}, height <= {height_threshold:.1f}")

        for box in all_boxes:
            paragraph = box['paragraph']
            text_content = self._extract_plain_text(paragraph)

            # STRICT CRITERIA - ALL must be met for furigana classification
            is_furigana = False
            reasons = []

            # MANDATORY: Must be very short (1-5 characters max)
            if len(text_content) > 5:
                regular_paragraphs.append(paragraph)
                continue

            # MANDATORY: Must be mostly/all hiragana (no kanji, minimal katakana)
            if not self._is_furigana_like_text(text_content):
                regular_paragraphs.append(paragraph)
                continue

            # MANDATORY: Must be small in size (both width AND height)
            is_small_width = box['width'] <= width_threshold and box['width'] > 0
            is_small_height = box['height'] <= height_threshold and box['height'] > 0

            if not (is_small_width or is_small_height):
                regular_paragraphs.append(paragraph)
                continue

            # Additional criteria that support furigana classification
            score = 0

            if is_small_width:
                score += 2
                reasons.append("small_width")

            if is_small_height:
                score += 2
                reasons.append("small_height")

            if len(text_content) <= 3:
                score += 2
                reasons.append("very_short")

            if self._is_all_hiragana(text_content):
                score += 1
                reasons.append("all_hiragana")

            # Width-to-height ratio for furigana
            if box['width'] > 0 and box['height'] > 0:
                ratio = box['width'] / box['height']
                if ratio > 1.5:  # Furigana is often wider than tall
                    score += 1
                    reasons.append("wide_ratio")

            # Need a minimum score to classify as furigana
            if score >= 3:
                is_furigana = True

            if is_furigana:
                furigana_paragraphs.append(paragraph)
                print(f"    FURIGANA: '{text_content}' (score: {score}) - {reasons}")
            else:
                regular_paragraphs.append(paragraph)

        print(f"  Furigana detection: {len(regular_paragraphs)} regular, {len(furigana_paragraphs)} furigana")
        return regular_paragraphs, furigana_paragraphs

    def _is_furigana_like_text(self, text: str) -> bool:
        """Check if text looks like furigana (mostly hiragana, no complex kanji)."""
        if not text:
            return False

        hiragana_count = 0
        kanji_count = 0
        katakana_count = 0
        other_count = 0

        for char in text:
            if self._is_hiragana(char):
                hiragana_count += 1
            elif self._is_kanji(char):
                kanji_count += 1
            elif self._is_katakana(char):
                katakana_count += 1
            elif self._is_japanese_char(char):
                other_count += 1

        total_japanese = hiragana_count + kanji_count + katakana_count

        if total_japanese == 0:
            return False

        # Furigana should be at least 80% hiragana and have no kanji
        hiragana_ratio = hiragana_count / total_japanese
        return hiragana_ratio >= 0.8 and kanji_count == 0

    def _calculate_paragraph_width(self, bounding_box) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 2:
            return 0

        try:
            x1 = vertices[0].x
            x2 = vertices[1].x
            return abs(x2 - x1)
        except (IndexError, AttributeError):
            return 0

    def _calculate_paragraph_height(self, bounding_box) -> float:
        """Calculate height of paragraph bounding box."""
        vertices = bounding_box.vertices
        if len(vertices) < 4:
            return 0

        try:
            y1 = vertices[0].y
            y3 = vertices[2].y  # Opposite corner
            return abs(y3 - y1)
        except (IndexError, AttributeError):
            return 0

    def _get_paragraph_y_position(self, bounding_box) -> float:
        """Get the Y position (top) of the paragraph."""
        vertices = bounding_box.vertices
        if not vertices:
            return 0

        try:
            return vertices[0].y
        except (IndexError, AttributeError):
            return 0

    def _extract_plain_text(self, paragraph) -> str:
        """Extract plain text from paragraph without confidence markers."""
        text_parts = []
        words = paragraph.words
        for word in words:
            symbols = word.symbols
            for symbol in symbols:
                text_parts.append(symbol.text)
        return ''.join(text_parts)

    def _is_mostly_hiragana(self, text: str) -> bool:
        """Check if text is mostly hiragana characters."""
        if not text:
            return False

        hiragana_count = 0
        total_chars = 0

        for char in text:
            if self._is_japanese_char(char):
                total_chars += 1
                if self._is_hiragana(char):
                    hiragana_count += 1

        if total_chars == 0:
            return False

        # Consider it mostly hiragana if 70% or more are hiragana
        return (hiragana_count / total_chars) >= 0.7

    def _is_all_hiragana(self, text: str) -> bool:
        """Check if text is all hiragana characters (ignoring non-Japanese chars)."""
        if not text:
            return False

        for char in text:
            if self._is_japanese_char(char) and not self._is_hiragana(char):
                return False

        return True

    def _is_hiragana(self, char: str) -> bool:
        """Check if character is hiragana."""
        return '\u3040' <= char <= '\u309F'

    def _is_katakana(self, char: str) -> bool:
        """Check if character is katakana."""
        return '\u30A0' <= char <= '\u30FF'

    def _is_kanji(self, char: str) -> bool:
        """Check if character is kanji."""
        return '\u4E00' <= char <= '\u9FAF'

    def _is_japanese_char(self, char: str) -> bool:
        """Check if character is Japanese (hiragana, katakana, or kanji)."""
        return self._is_hiragana(char) or self._is_katakana(char) or self._is_kanji(char)

    def _process_paragraph_text(self, paragraph) -> str:
        """Extract text from paragraph with confidence markers."""
        text_parts = []

        words = paragraph.words
        for word in words:
            symbols = word.symbols
            for symbol in symbols:
                text = symbol.text
                # Handle confidence from Vision object
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