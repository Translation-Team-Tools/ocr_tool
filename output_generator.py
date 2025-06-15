import datetime
from pathlib import Path
from typing import List, Dict


class OutputGenerator:
    """Generates final text output files."""

    def __init__(self):
        self.output_path = Path("result/ocr_output.txt")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_text_output(self, processed_results: List[Dict]):
        """Generate final text file with grouped results."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("=== Japanese OCR Results ===\n")
            f.write(f"Total Images Processed: {len(processed_results)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                "Confidence Markers: [?] = medium confidence, [??] = low confidence, [???] = very low confidence\n\n")

            for result in processed_results:
                relative_path = result['relative_path']
                f.write(f"=== Image: {relative_path} ===\n")

                # Write regular text
                for line in result['regular_text']:
                    if line.strip():  # Skip empty lines
                        f.write(f"[REGULAR] {line}\n")

                # Write furigana
                for line in result['furigana_text']:
                    if line.strip():  # Skip empty lines
                        f.write(f"[FURIGANA] {line}\n")

                f.write("\n")

        print(f"\nOutput generated: {self.output_path}")
