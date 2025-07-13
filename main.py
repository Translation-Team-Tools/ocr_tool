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


class OCRApplication:
    """Main application class for handling user interaction and workflow execution."""

    def __init__(self):
        self.project_root = project_root
        self.credentials_path = str(project_root / "credentials.json")

    def run(self) -> bool:
        """Main application entry point."""
        print("=" * 60)
        print("OCR Tool - Batch Image Text Recognition")
        print("=" * 60)

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
            print(f"\nProcessing images from: {input_folder}")
            print(f"Using credentials: {credentials_path}")
            print("-" * 60)

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
                self._show_failure_summary()

            return success

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return False
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            return False

    def _get_input_folder(self) -> str | None:
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

        # Show selected settings
        print(f"\nSelected settings:")
        print(f"  Quality: {quality.value}")
        print(f"  Max width: {settings.max_width}px")
        print(f"  JPEG quality: {settings.jpeg_quality}%")
        print(f"  Enhance contrast: {settings.enhance_contrast}")
        print(f"  Convert to grayscale: {settings.convert_to_grayscale}")

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

    def _validate_credentials(self) -> str | None:
        """Validate Google Cloud credentials."""
        print("\n3. Google Cloud Credentials")
        print("-" * 30)

        # Check default location
        if os.path.exists(self.credentials_path):
            print(f"Found credentials at: {self.credentials_path}")
            return self.credentials_path

        print(f"Credentials not found at default location: {self.credentials_path}")

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

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total images found: {summary['total']}")
        print(f"Successfully processed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")

        if summary['completed'] > 0:
            result_folder = Path(self.project_root) / "result"
            print(f"\nResults saved to: {result_folder}")
            print("- Optimized images in: optimized_images/")
            print("- Vision API responses in: vision_responses/")
            print("- Analysis results in: analysis_results/")

        print("\nProcessing complete!")

    def _show_failure_summary(self):
        """Show failure summary."""
        print("\n" + "=" * 60)
        print("PROCESSING FAILED")
        print("=" * 60)
        print("The OCR workflow encountered errors and could not complete.")
        print("Please check the error messages above for details.")


def main():
    """Application entry point."""
    app = OCRApplication()
    success = app.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())