from enum import Enum

class _Confidence:
    threshold: float
    marker: str

    def __init__(self, threshold: float, marker: str):
        self.threshold = threshold
        self.marker = marker

class Confidence(Enum):
    HIGH = _Confidence(threshold=0.90, marker='')
    MEDIUM = _Confidence(threshold=0.80, marker='[?]')
    LOW = _Confidence(threshold=0.60, marker='[??]')
    VERY_LOW = _Confidence(threshold=0.00, marker='[???]')