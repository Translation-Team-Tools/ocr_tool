#!/usr/bin/env python3
"""
OCR Tool Main Application
Handles user input and orchestrates the complete OCR workflow.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from core.img_processor import OptimizationSettings, OCRQuality
from workflow_manager import OCRWorkflowManager
from utils.logger import logger


class OCRApplication:
    """Main application class for handling user interaction and workflow execution."""

    def __init__(self):
        self.project_root = project_root
        self.credentials_path = str(project_root / "credentials.json")

    def run(self) -> bool:
        """Main application entry point."""
        logger.section_header("OCR TOOL - BATCH IMAGE TEXT RECOGNITION")

        try:
            # Get user inputs
            input_folder = self._get_input_folder()
            if not input_folder:
                return False

            optimization_settings = self._get_optimization_settings()
            credentials_path = self._validate_credentials()

            if not credentials_path:
                return False

            # Run workflow
            logger.info(f"Starting OCR processing: {Path(input_folder).name}")

            workflow_manager = OCRWorkflowManager(
                input_folder=input_folder,
                credentials_path=credentials_path,
                project_root=str(self.project_root)
            )

            success = workflow_manager.run_complete_workflow(optimization_settings)

            # Show summary
            if success:
                self._show_success_summary(workflow_manager)
            else:
                logger.error("Processing failed. Check error messages above.")

            return success

        except KeyboardInterrupt:
            logger.warning("Operation cancelled by user.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False

    def _get_input_folder(self) -> str:
        """Get and validate input folder from user."""
        print("\n1. Input Folder Selection")
        print("-" * 30)

        while True:
            input_folder = input("Enter path to image folder: ").strip()

            if not input_folder:
                print("Please enter a folder path.")
                continue

            # Expand user path and resolve
            input_folder = str(Path(input_folder).expanduser().resolve())

            if not os.path.exists(input_folder):
                print(f"Folder does not exist: {input_folder}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
                continue

            if not os.path.isdir(input_folder):
                print(f"Path is not a directory: {input_folder}")
                continue

            # Check if folder contains any image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            has_images = any(
                file.suffix.lower() in image_extensions
                for file in Path(input_folder).rglob('*')
                if file.is_file()
            )

            if not has_images:
                print(f"No supported image files found in: {input_folder}")
                print("Supported formats: JPG, JPEG, PNG")
                retry = input("Continue anyway? (y/n): ").strip().lower()
                if retry != 'y':
                    continue

            return input_folder

    def _get_optimization_settings(self) -> OptimizationSettings:
        """Get optimization settings from user."""
        print("\n2. Optimization Settings")
        print("-" * 30)

        # Quality setting
        print("Quality presets:")
        print("1. Fast (1600px max width, 85% JPEG quality)")
        print("2. Balanced (2048px max width, 90% JPEG quality) [Default]")
        print("3. Best (3000px max width, 95% JPEG quality)")

        while True:
            choice = input("Select quality (1-3) [2]: ").strip()
            if not choice:
                quality = OCRQuality.BALANCED
                break
            elif choice == '1':
                quality = OCRQuality.FAST
                break
            elif choice == '2':
                quality = OCRQuality.BALANCED
                break
            elif choice == '3':
                quality = OCRQuality.BEST
                break
            else:
                print("Please enter 1, 2, or 3.")

        # Enhancement options
        print("\nEnhancement options:")
        enhance_contrast = self._get_yes_no("Enhance contrast for better OCR? [Y/n]: ", default=True)
        convert_to_grayscale = self._get_yes_no("Convert to grayscale? [y/N]: ", default=False)

        settings = OptimizationSettings(
            quality=quality,
            enhance_contrast=enhance_contrast,
            convert_to_grayscale=convert_to_grayscale
        )

        # Show selected settings in one line
        print(
            f"\nUsing {quality.value} quality with {'contrast enhancement' if settings.enhance_contrast else 'no enhancement'}")

        return settings

    def _get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no input from user with default value."""
        while True:
            response = input(prompt).strip().lower()
            if not response:
                return default
            elif response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def _validate_credentials(self) -> str:
        """Validate Google Cloud credentials."""
        print("\n3. Google Cloud Credentials")
        print("-" * 30)

        # Check default location
        if os.path.exists(self.credentials_path):
            print(f"Using credentials: {Path(self.credentials_path).name}")
            return self.credentials_path

        print(f"Credentials not found at default location")

        while True:
            custom_path = input("Enter path to Google credentials JSON file: ").strip()

            if not custom_path:
                print("Credentials path is required.")
                continue

            custom_path = str(Path(custom_path).expanduser().resolve())

            if not os.path.exists(custom_path):
                print(f"File does not exist: {custom_path}")
                continue

            if not custom_path.endswith('.json'):
                print("Credentials file should be a JSON file.")
                continue

            try:
                # Basic validation - check if it's valid JSON
                import json
                with open(custom_path, 'r') as f:
                    json.load(f)
                return custom_path
            except json.JSONDecodeError:
                print("Invalid JSON file.")
                continue
            except Exception as e:
                print(f"Error reading file: {e}")
                continue

    def _show_success_summary(self, workflow_manager: OCRWorkflowManager):
        """Show success summary with processing statistics."""
        summary = workflow_manager.get_workflow_summary()
        input_folder_name = Path(workflow_manager.input_folder).name

        from rich.text import Text

        # Main success message with highlighted numbers
        text = Text()
        text.append("Processing complete! ")
        text.append(f"{summary['completed']}/{summary['total']}", style="bright_white")
        text.append(" images processed successfully")

        logger.console.print("✓ ", style="green", end="")
        logger.console.print(text)

        if summary['failed'] > 0 or summary['skipped'] > 0:
            logger.warning(f"Issues: {summary['failed']} failed, {summary['skipped']} skipped")

        result_folder = Path(self.project_root) / "result" / input_folder_name
        logger.info(f"Results saved to: {result_folder}")


def main():
    """Application entry point."""
    app = OCRApplication()
    success = app.run()

    # Add pause before closing - this prevents the console window from closing immediately
    print("\n" + "=" * 50)
    input("Press Enter to close...")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())