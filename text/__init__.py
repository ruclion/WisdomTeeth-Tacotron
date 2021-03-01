import re
from .symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}



def text_to_sequence(text):
  # map
  res =  _symbols_to_sequence(text)

  # Append EOS token
  res.append(_symbol_to_id['~'])
  return res


def sequence_to_text(sequence):
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      result += s
  return result



def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols]


