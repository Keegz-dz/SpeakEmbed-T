# SpeakEmbed-T Package
# This package provides voice cloning capabilities using neural speech synthesis

__version__ = "2.0.0"
__author__ = "SpeakEmbed-T Team"

# Import main components for easy access
from .scripts.main import Main
from .scripts.synthesizer import Synthesizer
from .scripts.vocoder import Vocoder
from .scripts.embed import Embed

__all__ = [
    "Main",
    "Synthesizer", 
    "Vocoder",
    "Embed"
]