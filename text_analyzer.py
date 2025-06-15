import statistics
from typing import Dict, Tuple, List


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

    def analyze_vision_response(self, vision_data: Dict) -> Dict:
        """Analyze Vision API response for text extraction and furigana detection."""
        # Handle both camelCase (MessageToDict) and snake_case (manual parsing) formats
        full_text_annotation = (
                vision_data.get('fullTextAnnotation') or  # MessageToDict format
                vision_data.get('full_text_annotation')  # Manual parsing format
        )

        if not full_text_annotation:
            print("  Warning: No text annotation found in response")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        # Extract all paragraphs
        paragraphs = []
        pages = full_text_annotation.get('pages', [])

        if not pages:
            print("  Warning: No pages found in text annotation")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        for page in pages:
            blocks = page.get('blocks', [])
            for block in blocks:
                block_paragraphs = block.get('paragraphs', [])
                paragraphs.extend(block_paragraphs)

        if not paragraphs:
            print("  Warning: No paragraphs found in pages")
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        print(f"  Found {len(paragraphs)} paragraphs to analyze")

        # Detect furigana based on bounding box width
        regular_paragraphs, furigana_paragraphs = self._detect_furigana(paragraphs)

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

    def _detect_furigana(self, paragraphs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Detect furigana based on bounding box width analysis."""
        if not paragraphs:
            return [], []

        # Calculate paragraph widths
        widths = []
        for paragraph in paragraphs:
            # Handle both camelCase and snake_case bounding box formats
            bounding_box = (
                    paragraph.get('boundingBox') or
                    paragraph.get('bounding_box')
            )
            if bounding_box:
                width = self._calculate_paragraph_width(bounding_box)
                widths.append(width)

        if not widths:
            # If no widths calculated, treat all as regular text
            return paragraphs, []

        # Use median width as baseline
        median_width = statistics.median(widths)
        furigana_threshold = median_width * 0.6  # 60% of normal width

        regular_paragraphs = []
        furigana_paragraphs = []

        for paragraph in paragraphs:
            bounding_box = (
                    paragraph.get('boundingBox') or
                    paragraph.get('bounding_box')
            )
            if bounding_box:
                width = self._calculate_paragraph_width(bounding_box)
                if width <= furigana_threshold:
                    furigana_paragraphs.append(paragraph)
                else:
                    regular_paragraphs.append(paragraph)
            else:
                # Default to regular if no bounding box
                regular_paragraphs.append(paragraph)

        print(f"  Furigana detection: {len(regular_paragraphs)} regular, {len(furigana_paragraphs)} furigana")
        return regular_paragraphs, furigana_paragraphs

    def _calculate_paragraph_width(self, bounding_box: Dict) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box.get('vertices', [])
        if len(vertices) < 2:
            return 0

        # Calculate width as distance between first two vertices
        try:
            x1 = vertices[0].get('x', 0)
            x2 = vertices[1].get('x', 0)
            return abs(x2 - x1)
        except (IndexError, TypeError):
            return 0

    def _process_paragraph_text(self, paragraph: Dict) -> str:
        """Extract text from paragraph with confidence markers."""
        text_parts = []

        words = paragraph.get('words', [])
        for word in words:
            symbols = word.get('symbols', [])
            for symbol in symbols:
                text = symbol.get('text', '')
                # Handle confidence as either float or string
                confidence_raw = symbol.get('confidence', 1.0)

                try:
                    if isinstance(confidence_raw, str):
                        confidence = float(confidence_raw)
                    else:
                        confidence = float(confidence_raw) if confidence_raw is not None else 1.0
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