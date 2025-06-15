import statistics
from typing import Dict, Tuple, List


class TextAnalyzer:
    """Analyzes OCR results for furigana detection and confidence marking."""

    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.95,
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
        if not vision_data.get('full_text_annotation'):
            return {'regular_paragraphs': [], 'furigana_paragraphs': []}

        # Extract all paragraphs
        paragraphs = []
        for page in vision_data['full_text_annotation']['pages']:
            for block in page['blocks']:
                for paragraph in block['paragraphs']:
                    paragraphs.append(paragraph)

        # Detect furigana based on bounding box width
        regular_paragraphs, furigana_paragraphs = self._detect_furigana(paragraphs)

        # Process text with confidence markers
        processed_regular = [self._process_paragraph_text(p) for p in regular_paragraphs]
        processed_furigana = [self._process_paragraph_text(p) for p in furigana_paragraphs]

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
            width = self._calculate_paragraph_width(paragraph['bounding_box'])
            widths.append(width)

        if not widths:
            return paragraphs, []

        # Use median width as baseline
        median_width = statistics.median(widths)
        furigana_threshold = median_width * 0.6  # 60% of normal width

        regular_paragraphs = []
        furigana_paragraphs = []

        for paragraph in paragraphs:
            width = self._calculate_paragraph_width(paragraph['bounding_box'])
            if width <= furigana_threshold:
                furigana_paragraphs.append(paragraph)
            else:
                regular_paragraphs.append(paragraph)

        return regular_paragraphs, furigana_paragraphs

    def _calculate_paragraph_width(self, bounding_box: Dict) -> float:
        """Calculate width of paragraph bounding box."""
        vertices = bounding_box['vertices']
        if len(vertices) < 2:
            return 0

        # Calculate width as distance between first two vertices
        return abs(vertices[1]['x'] - vertices[0]['x'])

    def _process_paragraph_text(self, paragraph: Dict) -> str:
        """Extract text from paragraph with confidence markers."""
        text_parts = []

        for word in paragraph.get('words', []):
            for symbol in word.get('symbols', []):
                text = symbol.get('text', '')
                confidence = symbol.get('confidence', 1.0)

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