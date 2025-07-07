import os
import tempfile
import webbrowser
from typing import List, Optional

from jinja2 import Environment


class OCRCharacter:
    """Represents a single character with confidence level"""

    def __init__(self, char: str, confidence: str = 'high'):
        self.char = char
        self.confidence = confidence  # 'high', 'medium', 'low', 'very_low'

    def to_dict(self):
        return {
            'char': self.char,
            'confidence': self.confidence
        }


class OCRLine:
    """Represents a line of text (regular or furigana)"""

    def __init__(self, is_furigana: bool = False):
        self.is_furigana = is_furigana
        self.characters: List[OCRCharacter] = []

    def add_character(self, char: str, confidence: str = 'high'):
        """Add a character to this line"""
        self.characters.append(OCRCharacter(char, confidence))
        return self

    def add_text(self, text: str, confidence: str = 'high'):
        """Add multiple characters with same confidence"""
        for char in text:
            self.add_character(char, confidence)
        return self

    def to_dict(self):
        return {
            'is_furigana': self.is_furigana,
            'characters': [char.to_dict() for char in self.characters]
        }


class OCRResult:
    """Represents OCR results for text analysis"""

    def __init__(self, title: str = ""):
        self.title = title
        self.lines: List[OCRLine] = []

    def add_line(self, is_furigana: bool = False) -> OCRLine:
        """Add a new line and return it for chaining"""
        line = OCRLine(is_furigana)
        self.lines.append(line)
        return line

    def add_text_line(self, text: str, confidence: str = 'high') -> OCRLine:
        """Add a regular text line with uniform confidence"""
        line = self.add_line(is_furigana=False)
        line.add_text(text, confidence)
        return line

    def add_furigana_line(self, text: str, confidence: str = 'high') -> OCRLine:
        """Add a furigana line with uniform confidence"""
        line = self.add_line(is_furigana=True)
        line.add_text(text, confidence)
        return line

    def to_dict(self):
        return {
            'title': self.title,
            'lines': [line.to_dict() for line in self.lines]
        }


class OCRHTMLGenerator:
    """Factory class for generating OCR HTML reports"""

    def __init__(self):
        self.results: List[OCRResult] = []
        self.main_title = "OCR Analysis Results"
        self.template_string = self._get_default_template()

    def add_result(self, title: str = "") -> OCRResult:
        """Add a new OCR result and return it for chaining"""
        result = OCRResult(title)
        self.results.append(result)
        return result

    def set_title(self, title: str):
        """Set the main title of the HTML report"""
        self.main_title = title
        return self

    def generate_html(self, output_path: Optional[str] = None, open_browser: bool = True) -> str:
        """Generate HTML and optionally save to file"""
        env = Environment()
        template = env.from_string(self.template_string)

        html_content = template.render(
            title=self.main_title,
            results=[result.to_dict() for result in self.results]
        )

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if open_browser:
                webbrowser.open(f'file://{os.path.abspath(output_path)}')
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                if open_browser:
                    webbrowser.open(f'file://{f.name}')

        return html_content

    def _get_default_template(self) -> str:
        """Returns the default HTML template from external file"""
        with open('html_template.html', 'r', encoding='utf-8') as f:
            return f.read()


# Example usage:
def demo_usage():
    """Demonstrate basic usage of the OCR HTML generator"""
    # Create generator
    generator = OCRHTMLGenerator()
    generator.set_title("Japanese OCR Analysis Results")

    # Add first result
    result1 = generator.add_result("Sample Text 1")

    # Add regular text line with mixed confidence
    line1 = result1.add_line(is_furigana=False)
    line1.add_character('今', 'high')
    line1.add_character('日', 'medium')
    line1.add_character('は', 'high')
    line1.add_character('良', 'low')
    line1.add_character('い', 'high')
    line1.add_character('天', 'medium')
    line1.add_character('気', 'very_low')
    line1.add_character('で', 'high')
    line1.add_character('す', 'high')
    line1.add_character('。', 'high')

    # Add furigana line
    furigana1 = result1.add_line(is_furigana=True)
    furigana1.add_character('き', 'high')
    furigana1.add_character('ょ', 'medium')
    furigana1.add_character('う', 'high')
    furigana1.add_character('よ', 'low')
    furigana1.add_character('い', 'high')
    furigana1.add_character('て', 'medium')
    furigana1.add_character('ん', 'very_low')
    furigana1.add_character('き', 'high')

    # Add second result
    result2 = generator.add_result("Sample Text 2")

    # Use convenience methods
    result2.add_text_line('学校に行きます。', 'high')
    result2.add_furigana_line('がっこう', 'medium')

    # Add challenging text
    difficult_line = result2.add_line(is_furigana=False)
    difficult_line.add_character('難', 'very_low')
    difficult_line.add_character('し', 'low')
    difficult_line.add_character('い', 'very_low')
    difficult_line.add_character('漢', 'medium')
    difficult_line.add_character('字', 'low')
    difficult_line.add_character('。', 'high')

    return generator


if __name__ == "__main__":
    # Demo usage
    demo_gen = demo_usage()
    html_content = demo_gen.generate_html(output_path="ocr_results.html")
    print("Generated OCR HTML report!")

    # Show basic usage
    print("\nBasic Usage:")
    print("generator = OCRHTMLGenerator()")
    print("result = generator.add_result('My Text')")
    print("line = result.add_line(is_furigana=False)")
    print("line.add_character('今', 'high')")
    print("generator.generate_html('output.html')")