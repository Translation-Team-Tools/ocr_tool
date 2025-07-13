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
            image = ImageProcessor._convert_to_rgb(image)

            # Apply grayscale conversion if requested
            if settings.convert_to_grayscale:
                image = image.convert('L').convert('RGB')  # Convert back to RGB for JPEG

            # Resize if image is too wide
            if image.width > settings.max_width:
                image = ImageProcessor._resize_image(image, settings.max_width)

            # Enhance contrast if requested
            if settings.enhance_contrast:
                image = ImageProcessor._enhance_contrast(image)

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

    @staticmethod
    def process_pil_image(image: PILImage.Image, settings: OptimizationSettings) -> PILImage.Image:
        """
        Process PIL Image object according to optimization settings.

        Args:
            image: PIL Image object
            settings: Optimization settings

        Returns:
            Processed PIL Image object
        """
        try:
            # Convert to RGB if necessary (handles various formats)
            image = ImageProcessor._convert_to_rgb(image)

            # Apply grayscale conversion if requested
            if settings.convert_to_grayscale:
                image = image.convert('L').convert('RGB')  # Convert back to RGB for JPEG

            # Resize if image is too large
            if max(image.size) > settings.max_width:
                image = ImageProcessor._resize_image_by_max_dimension(image, settings.max_width)

            # Enhance contrast if requested
            if settings.enhance_contrast:
                image = ImageProcessor._enhance_contrast(image)

            return image

        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")

    @staticmethod
    def _convert_to_rgb(image: PILImage.Image) -> PILImage.Image:
        """Convert image to RGB format, handling transparency properly."""
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB, handling transparency
            rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                if 'transparency' in image.info or len(image.split()) > 3:
                    rgb_image.paste(image, mask=image.split()[-1])
                else:
                    rgb_image.paste(image)
            else:
                rgb_image.paste(image)
            return rgb_image
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image

    @staticmethod
    def _resize_image(image: PILImage.Image, max_width: int) -> PILImage.Image:
        """Resize image maintaining aspect ratio with max width constraint."""
        aspect_ratio = image.height / image.width
        new_height = int(max_width * aspect_ratio)
        return image.resize((max_width, new_height), PILImage.Resampling.LANCZOS)

    @staticmethod
    def _resize_image_by_max_dimension(image: PILImage.Image, max_dimension: int) -> PILImage.Image:
        """Resize image maintaining aspect ratio with max dimension constraint."""
        ratio = max_dimension / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, PILImage.Resampling.LANCZOS)

    @staticmethod
    def _enhance_contrast(image: PILImage.Image, factor: float = 1.2) -> PILImage.Image:
        """Enhance image contrast for better OCR results."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)