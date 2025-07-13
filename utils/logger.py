import sys
import time
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


class VisualLogger:
    """Enhanced logger with colors and dynamic status updates."""

    def __init__(self):
        self.current_status = ""
        self.status_line_active = False

    def clear_status_line(self):
        """Clear the current status line."""
        if self.status_line_active:
            # Move cursor up one line and clear it
            sys.stdout.write('\033[A\033[K')
            sys.stdout.flush()
            self.status_line_active = False

    def log(self, message: str, level: LogLevel = LogLevel.INFO, indent: int = 0):
        """Log a message with color coding."""
        self.clear_status_line()

        # Choose color and icon based on level
        if level == LogLevel.SUCCESS:
            color = Colors.BRIGHT_GREEN
            icon = "✓"
        elif level == LogLevel.ERROR:
            color = Colors.BRIGHT_RED
            icon = "✗"
        elif level == LogLevel.WARNING:
            color = Colors.BRIGHT_YELLOW
            icon = "⚠"
        elif level == LogLevel.PROGRESS:
            color = Colors.BRIGHT_CYAN
            icon = "→"
        else:  # INFO
            color = Colors.BRIGHT_BLUE
            icon = "→"

        # Create indentation
        indent_str = "  " * indent

        # Print colored message
        print(f"{indent_str}{color}{icon} {message}{Colors.RESET}")
        sys.stdout.flush()

    def status(self, message: str, indent: int = 0):
        """Update status line in place (overwrites previous status)."""
        self.clear_status_line()

        indent_str = "  " * indent
        status_msg = f"{indent_str}{Colors.BRIGHT_CYAN}⟳ {message}...{Colors.RESET}"

        print(status_msg)
        sys.stdout.flush()
        self.status_line_active = True
        self.current_status = message

    def progress(self, current: int, total: int, item_name: str = "", indent: int = 0):
        """Show progress with a visual progress bar."""
        self.clear_status_line()

        # Calculate percentage
        percentage = (current / total) * 100 if total > 0 else 0

        # Create progress bar (20 characters wide)
        bar_width = 20
        filled = int((current / total) * bar_width) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format message
        indent_str = "  " * indent
        if item_name:
            message = f"{indent_str}{Colors.BRIGHT_CYAN}⟳ Processing [{bar}] {current}/{total} ({percentage:.0f}%) - {item_name}{Colors.RESET}"
        else:
            message = f"{indent_str}{Colors.BRIGHT_CYAN}⟳ Progress [{bar}] {current}/{total} ({percentage:.0f}%){Colors.RESET}"

        print(message)
        sys.stdout.flush()
        self.status_line_active = True

    def step_header(self, step_number: int, title: str, total_steps: int = None):
        """Print a major step header."""
        self.clear_status_line()

        if total_steps:
            step_info = f"Step {step_number}/{total_steps}"
        else:
            step_info = f"Step {step_number}"

        print(f"\n{Colors.BOLD}{Colors.BG_BLUE} {step_info}: {title} {Colors.RESET}")
        sys.stdout.flush()

    def section_header(self, title: str):
        """Print a section header."""
        self.clear_status_line()
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{title.center(60)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{'=' * 60}{Colors.RESET}")
        sys.stdout.flush()

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
        full_message = f"{message} ({Colors.DIM}{timing_str}{Colors.RESET})"
        self.success(full_message, indent)

    def size_info(self, filename: str, size: int, indent: int = 0):
        """Log file size information."""
        size_str = self.format_size(size)
        self.info(f"{filename} - {Colors.DIM}{size_str}{Colors.RESET}", indent)

    def compression_info(self, original_size: int, optimized_size: int, indent: int = 0):
        """Log compression information."""
        reduction = ((original_size - optimized_size) / original_size) * 100 if original_size > 0 else 0
        original_str = self.format_size(original_size)
        optimized_str = self.format_size(optimized_size)

        if reduction > 0:
            color = Colors.BRIGHT_GREEN
        else:
            color = Colors.YELLOW

        message = f"Size: {original_str} → {optimized_str} {color}({reduction:.1f}% reduction){Colors.RESET}"
        self.info(message, indent)

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