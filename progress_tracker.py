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

        if image_num > 0:
            avg_time = elapsed / image_num
            remaining = (self.total_images - image_num) * avg_time
            eta = self._format_time(remaining)
        else:
            eta = "calculating..."

        folder = os.path.dirname(image_path)
        filename = os.path.basename(image_path)

        print(f"Processing: {folder}/{filename} ({image_num}/{self.total_images}) "
              f"Stage {self.current_stage}/{self.total_stages} - ETA: {eta}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
