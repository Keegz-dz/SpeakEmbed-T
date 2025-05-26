"""
Text processing utilities for converting between text and symbol/ID sequences for TTS models.

This module is adapted from the Tacotron/Tacotron 2 text processing pipeline.
Original authors: NVIDIA, Keith Ito, and contributors (https://github.com/keithito/tacotron)
Modifications may have been made for this project.
"""

import re
from utils.symbols import symbols
from . import cleaners

# -----------------------------------------------------------------------------
# Symbol <-> ID mappings
# -----------------------------------------------------------------------------

# Map each symbol to a unique numeric ID and vice versa
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# -----------------------------------------------------------------------------
# Regular expressions
# -----------------------------------------------------------------------------

# Matches text enclosed in curly braces (used for ARPAbet sequences)
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def text_to_sequence(text, cleaner_names):
    """
    Converts a string of text to a sequence of symbol IDs.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
        text (str): Input string to convert to a sequence.
        cleaner_names (list of str): Names of cleaner functions to run the text through.

    Returns:
        List[int]: Sequence of symbol IDs corresponding to the input text.
    """
    sequence = []

    # Process curly-brace ARPAbet sequences separately
    while len(text):
        m = _curly_re.match(text)
        if not m:
            # No more curly braces; clean and convert the rest
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        # Clean and convert text before the curly braces
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        # Convert ARPAbet inside curly braces
        sequence += _arpabet_to_sequence(m.group(2))
        # Continue with the rest of the text
        text = m.group(3)

    # Append EOS (end-of-sequence) token
    sequence.append(_symbol_to_id["~"])
    return sequence


def sequence_to_text(sequence):
    """
    Converts a sequence of symbol IDs back to a string.

    Args:
        sequence (List[int]): Sequence of symbol IDs.

    Returns:
        str: The corresponding text string.
    """
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet symbols back in curly braces
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    # Replace adjacent ARPAbet curly braces with a space
    return result.replace("}{", " ")

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _clean_text(text, cleaner_names):
    """
    Applies a sequence of cleaner functions to the input text.
    """
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    """
    Converts a list of symbols to their corresponding sequence of IDs.
    """
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    """
    Converts an ARPAbet string (space-separated) to a sequence of IDs, prefixing each with '@'.
    """
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    """
    Determines if a symbol should be included in the sequence (filters out padding and EOS).
    """
    return s in _symbol_to_id and s not in ("_", "~")
