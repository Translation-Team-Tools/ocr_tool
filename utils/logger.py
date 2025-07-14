import time
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text


class LogLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"


class VisualLogger:
    """Enhanced logger using Rich for better terminal compatibility."""

    def __init__(self):
        self.console = Console()
        self.current_status = ""
        self.status_line_active = False
        self._live_display = None
        self._last_progress_text = ""

    def clear_status_line(self):
        """Clear the current status line."""
        if self._live_display:
            self._live_display.stop()
            self._live_display = None
        self.status_line_active = False

    def log(self, message: str, level: LogLevel = LogLevel.INFO, indent: int = 0):
        """Log a message with subtle color coding using Rich."""
        # Clear any active live display first
        self.clear_status_line()

        # Choose color and icon based on level
        if level == LogLevel.SUCCESS:
            icon_style = "green"
            icon = "✓"
        elif level == LogLevel.ERROR:
            icon_style = "red"
            icon = "✗"
        elif level == LogLevel.WARNING:
            icon_style = "yellow"
            icon = "⚠"
        elif level == LogLevel.PROGRESS:
            icon_style = "cyan"
            icon = "→"
        else:  # INFO
            icon_style = "blue"
            icon = "→"

        # Create indentation
        indent_str = "  " * indent

        # Print message with colored icon only
        from rich.text import Text
        text = Text()
        text.append(f"{indent_str}")
        text.append(f"{icon} ", style=icon_style)
        text.append(message)

        self.console.print(text)

    def status(self, message: str, indent: int = 0):
        """Update status line using Rich."""
        self.clear_status_line()
        from rich.text import Text

        indent_str = "  " * indent
        text = Text()
        text.append(f"{indent_str}")
        text.append("⟳ ", style="cyan")
        text.append(f"{message}...")

        self.console.print(text)
        self.status_line_active = True
        self.current_status = message

    def progress(self, current: int, total: int, item_name: str = "", indent: int = 0):
        """Show progress with a visual progress bar that updates in place."""
        from rich.text import Text

        # Calculate percentage
        percentage = (current / total) * 100 if total > 0 else 0

        # Create progress bar (20 characters wide)
        bar_width = 20
        filled = int((current / total) * bar_width) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Create styled text
        indent_str = "  " * indent
        progress_text = Text()
        progress_text.append(f"{indent_str}")
        progress_text.append("⟳ ", style="cyan")
        progress_text.append("Processing [")
        progress_text.append(bar, style="cyan")
        progress_text.append("] ")
        progress_text.append(f"{current}/{total}", style="bright_white")
        progress_text.append(f" ({percentage:.0f}%)")

        if item_name:
            progress_text.append(" - ")
            progress_text.append(item_name, style="dim")

        # If this is the first progress call or we're not in live mode, start live display
        if not self._live_display:
            self._live_display = Live(
                progress_text,
                console=self.console,
                refresh_per_second=10
            )
            self._live_display.start()
        else:
            # Update the existing live display
            self._live_display.update(progress_text)

        self.status_line_active = True

    def progress_complete(self, message: str = None):
        """Complete the progress display and optionally show a final message."""
        if self._live_display:
            self._live_display.stop()
            self._live_display = None

        if message:
            self.success(message)

        self.status_line_active = False

    def progress_bar_context(self, items, description="Processing"):
        """Create a Rich progress bar context manager for more advanced usage."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )

    def step_header(self, step_number: int, title: str, total_steps: int = None):
        """Print a major step header using Rich panels."""
        # Clear any active progress display
        self.clear_status_line()

        if total_steps:
            step_info = f"Step {step_number}/{total_steps}"
        else:
            step_info = f"Step {step_number}"

        panel = Panel(
            f"[white]{step_info}: {title}[/white]",
            style="dim blue",
            padding=(0, 1)
        )
        self.console.print()
        self.console.print(panel)

    def section_header(self, title: str):
        """Print a section header using Rich."""
        self.clear_status_line()
        self.console.print()
        self.console.rule(f"[white]{title}[/white]", style="dim white")

    def info(self, message: str, indent: int = 0):
        """Log info message."""
        self.log(message, LogLevel.INFO, indent)

    def success(self, message: str, indent: int = 0):
        """Log success message."""
        self.log(message, LogLevel.SUCCESS, indent)

    def warning(self, message: str, indent: int = 0):
        """Log warning message."""
        self.log(message, LogLevel.WARNING, indent)

    def error(self, message: str, indent: int = 0):
        """Log error message."""
        self.log(message, LogLevel.ERROR, indent)

    def timing(self, message: str, duration: float, indent: int = 0):
        """Log a message with timing information."""
        self.clear_status_line()
        from rich.text import Text

        timing_str = f"{duration:.1f}s"
        indent_str = "  " * indent

        text = Text()
        text.append(f"{indent_str}")
        text.append("✓ ", style="green")
        text.append(message)
        text.append(f" ({timing_str})", style="dim")

        self.console.print(text)

    def size_info(self, filename: str, size: int, indent: int = 0):
        """Log file size information."""
        self.clear_status_line()
        from rich.text import Text

        size_str = self.format_size(size)
        indent_str = "  " * indent

        text = Text()
        text.append(f"{indent_str}")
        text.append("→ ", style="blue")
        text.append(filename)
        text.append(f" - {size_str}", style="dim")

        self.console.print(text)

    def compression_info(self, original_size: int, optimized_size: int, indent: int = 0):
        """Log compression information."""
        self.clear_status_line()
        from rich.text import Text

        reduction = ((original_size - optimized_size) / original_size) * 100 if original_size > 0 else 0
        original_str = self.format_size(original_size)
        optimized_str = self.format_size(optimized_size)

        color = "green" if reduction > 0 else "yellow"
        indent_str = "  " * indent

        text = Text()
        text.append(f"{indent_str}")
        text.append("→ ", style="blue")
        text.append(f"Size: {original_str} → {optimized_str}")
        text.append(f" ({reduction:.1f}% reduction)", style=color)

        self.console.print(text)

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"


# Global logger instance
logger = VisualLogger()

# Example usage
if __name__ == "__main__":
    logger.section_header("Image Processing Demo")
    logger.step_header(1, "Loading Images", 3)

    # Test the progress method that updates in place
    for i in range(4):
        logger.progress(i + 1, 4, f"image_{i + 1}.jpg")
        time.sleep(0.5)

    logger.progress_complete("All images loaded successfully")
    logger.timing("Total processing time", 2.1)