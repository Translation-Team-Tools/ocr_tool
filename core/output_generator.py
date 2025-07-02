from datetime import datetime
from typing import List


class OutputGenerator:
    """Generates formatted text output from analyzed OCR results."""

    def _build_header(self, total_images:int) -> str:
        """Build the header section."""
        header_lines = [
            "=== Japanese OCR Results ===",
            f"Total Images Processed: {total_images}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Confidence Markers: [?] = medium confidence, [??] = low confidence, [???] = very low confidence"
        ]
        return "\n".join(header_lines)

    def build_image_section(self, lines: List[str], filename:str) -> str:
        """Build the section for a single image."""
        lines.insert(0, f"=== Image: {filename} ===")

        return "\n".join(lines)

    def build_final_result(self, image_sections: List[str]):
        result = self._build_header(total_images=len(image_sections)) +"\n"

        for img_section in image_sections:
            result+= f"\n\n{img_section}\n\n"

        return result

    def mark_char(self, text:str, marker:str):
        return f"{marker}{text}{marker}"

    def build_line(self, text:str, is_furigana:bool):
        return f"{'[FURIGANA]' if is_furigana else '[REGULAR]'} {text}"