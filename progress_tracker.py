import os
import time


class ProgressTracker:
    """Tracks and displays processing progress with ETA calculations."""

    def __init__(self, total_images: int):
        self.total_images = total_images
        self.current_image = 0
        self.current_stage = 1
        self.total_stages = 5
        self.stage_names = {
            1: "Discovering Files",
            2: "Copying Images",
            3: "OCR Processing",
            4: "Analyzing Results",
            5: "Generating Output"
        }
        self.start_time = time.time()
        self.stage_start_time = time.time()

    def update_stage(self, stage_num: int):
        """Update current processing stage."""
        self.current_stage = stage_num
        self.stage_start_time = time.time()
        print(f"\n=== Stage {stage_num}/{self.total_stages}: {self.stage_names[stage_num]} ===")

    def update_image(self, image_num: int, image_path: str):
        """Update current image being processed with ETA calculation."""
        self.current_image = image_num
        elapsed = time.time() - self.start_time

        folder = os.path.dirname(image_path)
        filename = os.path.basename(image_path)

        print(f"Processing: {folder}/{filename} ({image_num}/{self.total_images} images)")
