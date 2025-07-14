import time
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


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

    def clear_status_line(self):
        """Clear the current status line."""
        # Rich handles this automatically, so we don't need to do anything
        pass

    def log(self, message: str, level: LogLevel = LogLevel.INFO, indent: int = 0):
        """Log a message with color coding using Rich."""

        # Choose color and icon based on level
        if level == LogLevel.SUCCESS:
            style = "bold green"
            icon = "✓"
        elif level == LogLevel.ERROR:
            style = "bold red"
            icon = "✗"
        elif level == LogLevel.WARNING:
            style = "bold yellow"
            icon = "⚠"
        elif level == LogLevel.PROGRESS:
            style = "bold cyan"
            icon = "→"
        else:  # INFO
            style = "bold blue"
            icon = "→"

        # Create indentation
        indent_str = "  " * indent

        # Print colored message
        self.console.print(f"{indent_str}{icon} {message}", style=style)

    def status(self, message: str, indent: int = 0):
        """Update status line using Rich."""
        indent_str = "  " * indent
        self.console.print(f"{indent_str}⟳ {message}...", style="bold cyan")
        self.status_line_active = True
        self.current_status = message

    def progress(self, current: int, total: int, item_name: str = "", indent: int = 0):
        """Show progress with a visual progress bar - matches original API."""
        # Calculate percentage
        percentage = (current / total) * 100 if total > 0 else 0

        # Create progress bar (20 characters wide)
        bar_width = 20
        filled = int((current / total) * bar_width) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format message
        indent_str = "  " * indent
        if item_name:
            message = f"{indent_str}⟳ Processing [{bar}] {current}/{total} ({percentage:.0f}%) - {item_name}"
        else:
            message = f"{indent_str}⟳ Progress [{bar}] {current}/{total} ({percentage:.0f}%)"

        self.console.print(message, style="bold cyan")
        self.status_line_active = True

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
        if total_steps:
            step_info = f"Step {step_number}/{total_steps}"
        else:
            step_info = f"Step {step_number}"

        panel = Panel(
            f"[bold white]{step_info}: {title}[/bold white]",
            style="blue",
            padding=(0, 1)
        )
        self.console.print()
        self.console.print(panel)

    def section_header(self, title: str):
        """Print a section header using Rich."""
        self.console.print()
        self.console.rule(f"[bold white]{title}[/bold white]", style="white")

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
        timing_str = f"{duration:.1f}s"
        indent_str = "  " * indent
        # Print success message with timing info
        self.console.print(f"{indent_str}✓ {message}", style="bold green", end="")
        self.console.print(f" [dim]({timing_str})[/dim]")

    def size_info(self, filename: str, size: int, indent: int = 0):
        """Log file size information."""
        size_str = self.format_size(size)
        indent_str = "  " * indent
        self.console.print(f"{indent_str}→ {filename}", style="bold blue", end="")
        self.console.print(f" [dim]- {size_str}[/dim]")

    def compression_info(self, original_size: int, optimized_size: int, indent: int = 0):
        """Log compression information."""
        reduction = ((original_size - optimized_size) / original_size) * 100 if original_size > 0 else 0
        original_str = self.format_size(original_size)
        optimized_str = self.format_size(optimized_size)

        style = "bold green" if reduction > 0 else "yellow"

        indent_str = "  " * indent
        self.console.print(
            f"{indent_str}→ Size: {original_str} → {optimized_str}",
            style="bold blue",
            end=""
        )
        self.console.print(f" [{style}]({reduction:.1f}% reduction)[/{style}]")

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

    # Test the progress method that matches your original API
    for i in range(4):
        logger.progress(i + 1, 4, f"image_{i + 1}.jpg")
        time.sleep(0.2)

    logger.success("All images loaded successfully")
    logger.timing("Total processing time", 2.1)
    logger.compression_info(1024000, 512000)