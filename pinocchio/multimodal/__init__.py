"""Pinocchio multimodal processor modules."""

from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.multimodal.audio_processor import AudioProcessor
from pinocchio.multimodal.video_processor import VideoProcessor

__all__ = [
    "TextProcessor",
    "VisionProcessor",
    "AudioProcessor",
    "VideoProcessor",
]
