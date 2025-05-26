"""
Defines the set of symbols used in text input to the model.

Default is a set of ASCII characters for English or text run through Unidecode.
Adapted from the Tacotron project by Keith Ito and contributors.
"""
# from . import cmudict

_pad        = "_"
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet
