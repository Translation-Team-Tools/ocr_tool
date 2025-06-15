from typing import List


class ErrorHandler:
    """Handles and categorizes different types of processing errors."""

    def __init__(self):
        self.failed_images = []
        self.error_types = {
            'network': [],
            'api_quota': [],
            'file_access': [],
            'vision_processing': []
        }

    def add_error(self, image_path: str, error_type: str, error_message: str):
        """Add error to tracking system."""
        error_info = {
            'path': image_path,
            'type': error_type,
            'message': error_message
        }
        self.failed_images.append(error_info)
        self.error_types[error_type].append(image_path)

    def categorize_error(self, exception: Exception) -> str:
        """Categorize error based on exception type and message."""
        error_str = str(exception).lower()

        if 'quota' in error_str or 'limit' in error_str:
            return 'api_quota'
        elif 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return 'network'
        elif 'permission' in error_str or 'access' in error_str:
            return 'file_access'
        else:
            return 'vision_processing'

    def prompt_retry_failed(self) -> List[str]:
        """Prompt user about retrying failed images."""
        if not self.failed_images:
            return []

        print(f"\n{len(self.failed_images)} images failed to process:")
        for error in self.failed_images:
            print(f"  - {error['path']}: {error['message']}")

        response = input("\nRetry failed images? (y/N): ").lower().strip()
        if response == 'y':
            return [error['path'] for error in self.failed_images]
        else:
            print("Skipping failed images. They will not be included in output.")
            return []
