from dataclasses import dataclass
from enum import Enum
from io import BytesIO

from PIL import Image as PILImage, ImageEnhance


class OCRQuality(Enum):
    """Simple quality options for OCR optimization."""
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


@dataclass
class OptimizationSettings:
    """OCR optimization settings with user-friendly options."""
    quality: OCRQuality = OCRQuality.BALANCED
    enhance_contrast: bool = True
    convert_to_grayscale: bool = False
    max_width: int = 2048
    jpeg_quality: int = 90

    def __post_init__(self):
        """Adjust settings based on quality preset."""
        if self.quality == OCRQuality.FAST:
            self.max_width = 1600
            self.jpeg_quality = 85
        elif self.quality == OCRQuality.BEST:
            self.max_width = 3000
            self.jpeg_quality = 95


class ImageProcessor:
    @staticmethod
    def process_image(image_bytes: bytes, settings: OptimizationSettings) -> bytes:
        """
        Process image bytes according to optimization settings.

        Args:
            image_bytes: Raw image bytes
            settings: Optimization settings

        Returns:
            Processed image bytes
        """
        try:
            # Load image from bytes
            with BytesIO(image_bytes) as input_buffer:
                image = PILImage.open(input_buffer)
                image.load()  # Ensure image is fully loaded

            # Convert to RGB if necessary (handles various formats)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB, handling transparency
                rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                if image.mode in ('RGBA', 'LA'):
                    rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply grayscale conversion if requested
            if settings.convert_to_grayscale:
                image = image.convert('L').convert('RGB')  # Convert back to RGB for JPEG

            # Resize if image is too wide
            if image.width > settings.max_width:
                # Calculate new height maintaining aspect ratio
                aspect_ratio = image.height / image.width
                new_height = int(settings.max_width * aspect_ratio)
                image = image.resize((settings.max_width, new_height), PILImage.Resampling.LANCZOS)

            # Enhance contrast if requested
            if settings.enhance_contrast:
                enhancer = ImageEnhance.Contrast(image)
                # Slightly increase contrast (1.2x) - good for OCR
                image = enhancer.enhance(1.2)

            # Save optimized image to bytes
            output_buffer = BytesIO()
            image.save(
                output_buffer,
                format='JPEG',
                quality=settings.jpeg_quality,
                optimize=True
            )
            return output_buffer.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")