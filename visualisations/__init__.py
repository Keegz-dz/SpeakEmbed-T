"""Lightweight visualisations package init with lazy imports.

Avoid importing heavy optional dependencies (e.g., visdom, umap) at package import time.
"""

__all__ = ["plot_spectrogram", "plot_waveform", "interactive_diarization"]

def __getattr__(name):
    if name in ("plot_spectrogram", "plot_waveform"):
        from .visualizations import plot_spectrogram, plot_waveform
        return {"plot_spectrogram": plot_spectrogram, "plot_waveform": plot_waveform}[name]
    if name == "interactive_diarization":
        from .speaker_diarisation_utils import interactive_diarization
        return interactive_diarization
    raise AttributeError(name)