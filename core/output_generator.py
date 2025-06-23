from datetime import datetime
from pathlib import Path
from typing import List, Dict


class OutputGenerator:
    """Generates formatted text output from analyzed OCR results."""

    def __init__(self, output_folder: str = "result"):
        self.output_folder = Path(output_folder)
        self.output_file = self.output_folder / "ocr_output.txt"

    def generate_output(self, analyzed_results: List[Dict]) -> None:
        """Generate formatted text output file from analyzed results."""
        if not analyzed_results:
            print("No results to generate output")
            return

        # Ensure output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Generate content
        content = self._build_output_content(analyzed_results)

        # Write to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Output written to: {self.output_file}")
        print(f"Total images in output: {len(analyzed_results)}")

    def _build_output_content(self, analyzed_results: List[Dict]) -> str:
        """Build the complete output content."""
        lines = []

        # Header
        lines.extend(self._build_header(analyzed_results))
        lines.append("")

        # Per-image sections
        for result in analyzed_results:
            lines.extend(self._build_image_section(result))
            lines.append("")

        return "\n".join(lines)

    def _build_header(self, analyzed_results: List[Dict]) -> List[str]:
        """Build the header section."""
        header_lines = [
            "=== Japanese OCR Results ===",
            f"Total Images Processed: {len(analyzed_results)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Confidence Markers: [?] = medium confidence, [??] = low confidence, [???] = very low confidence"
        ]
        return header_lines

    def _build_image_section(self, result: Dict) -> List[str]:
        """Build the section for a single image."""
        lines = []

        # Image header
        lines.append(f"=== Image: {result['filename']} ===")

        # Regular text section
        if result['regular_paragraphs']:
            for paragraph in result['regular_paragraphs']:
                lines.append(f"[REGULAR] {paragraph}")

        # Furigana text section
        if result['furigana_paragraphs']:
            for paragraph in result['furigana_paragraphs']:
                lines.append(f"[FURIGANA] {paragraph}")

        return lines