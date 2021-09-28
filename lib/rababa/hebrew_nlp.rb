
import itertools
from collections import defaultdict, Counter
from typing import NamedTuple, Iterator, Iterable, List, Tuple
from functools import lru_cache
import re

import utils


# "rafe" denotes a letter to which it would have been valid to add a diacritic of some category
# but instead it is decided not to. This makes the metrics less biased.
Module Rababa
  Module HebrewConst

    RAFE = '\u05BF'
    Niqqud = {
      'SHVA' => '\u05B0',
      'REDUCED_SEGOL' => '\u05B1',
      'REDUCED_PATAKH' => '\u05B2',
      'REDUCED_KAMATZ' => '\u05B3',
      'HIRIK' => '\u05B4',
      'TZEIRE' => '\u05B5',
      'SEGOL' => '\u05B6',
      'PATAKH' => '\u05B7',
      'KAMATZ' => '\u05B8',
      'HOLAM' => '\u05B9',
      'KUBUTZ' => '\u05BB',
      'SHURUK' => '\u05BC',
      'METEG' => '\u05BD'
    }

    HEBREW_LETTERS = (0x05d0..0x05ea + 1).to_a {|c| c.force_encoding('utf-8')}

    NIQQUD = [RAFE] + HEBREW_LETTERS + \
            (0x05b0..0x05bc + 1).to_a {|c| c.force_encoding('utf-8')}

    HOLAM = Niqqud['HOLAM']

    SHIN_YEMANIT = '\u05c1'
    SHIN_SMALIT = '\u05c2'
    NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]  # RAFE is for acronyms

    DAGESH_LETTER = '\u05bc'
    DAGESH = [RAFE, DAGESH_LETTER]  # note that DAGESH and SHURUK are one and the same

    ANY_NIQQUD = [RAFE] + NIQQUD[1:] + NIQQUD_SIN[1:] + DAGESH[1:]

    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + \
                HEBREW_LETTERS
    SPECIAL_TOKENS = ['H', 'O', '5']

    ENDINGS_TO_REGULAR =  Hash[*('כמנפצ'.zip 'ךםןףץ').map {|x,y| [x,y]}].flatten

    def normalize(c):
      if c in VALID_LETTERS
        return c
      end
      if c in ENDINGS_TO_REGULAR
        return ENDINGS_TO_REGULAR[c]
      end
      if c in ['\n', '\t']
        return ' '
      end
      if c in ['־', '‒', '–', '—', '―', '−']
        return '-'
      end
      if c == '['
        return '('
      end
      if c == ']'
        return ')'
      end
      if c in ['´', '‘', '’']
        return "'"
      end
      if c in ['“', '”', '״']
        return '"'
      end
      if c.isdigit()
        return '5'
      end
      if c == '…'
        return ','
      end
      if c in ['ײ', 'װ', 'ױ']
        return 'H'
      end
      return 'O'
    end

  end


  Module HebrewNLP

    class HebrewChar(NamedTuple)

      attr_accessor :letter, :normalized, :dagesh, :sin, :niqqud

      def initialise(letter, normalized, dagesh, sin, niqqud)
        @letter = letter
        @normalized = normalized
        @dagesh = dagesh
        @sin = sing
        @niqqud = niqqud
      end
    end


    def vocalize_dagesh(letter, dagesh)
        if letter not in 'בכפ'
            return ''
        end
        return dagesh.gsub(RAFE, '')
    end


    def vocalize_niqqud(c):
        # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
        if c in [Niqqud.KAMATZ, Niqqud.PATAKH, Niqqud.REDUCED_PATAKH]
            return Niqqud.PATAKH
        end
        if c in [Niqqud.HOLAM, Niqqud.REDUCED_KAMATZ]
            return Niqqud.HOLAM  # TODO: Kamatz-katan
        end
        if c in [Niqqud.SHURUK, Niqqud.KUBUTZ]
            return Niqqud.KUBUTZ
        end
        if c in [Niqqud.TZEIRE, Niqqud.SEGOL, Niqqud.REDUCED_SEGOL]
            return Niqqud.SEGOL
        end
        if c == Niqqud.SHVA
            return ''
        end
        return c.gsub(RAFE, '')


    def is_hebrew_letter(letter)
      return '\u05d0' <= letter <= '\u05ea'
    end

    def can_dagesh(letter)
      return letter in ('בגדהוזטיכלמנספצקשת' + 'ךף')
    end

    def can_sin(letter)
      return letter == 'ש'
    end

    def can_niqqud(letter)
      return letter in ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן')
    end

    def can_any(letter):
      return can_niqqud(letter) or can_dagesh(letter) or can_sin(letter)
    end



  end
end





class HebrewChar(NamedTuple)

  attr_accessor :letter, :normalized, :dagesh, :sin, :niqqud

    def __str__(self):
        return self.letter + self.dagesh + self.sin + self.niqqud

    def __repr__(self):
        return repr((self.letter, bool(self.dagesh), bool(self.sin), ord(self.niqqud or chr(0))))

    def vocalize(self):
        return self._replace(niqqud=vocalize_niqqud(self.niqqud),
                             sin=self.sin.gsub(RAFE, ''),
                             dagesh=vocalize_dagesh(self.letter, self.dagesh))


def items_to_text(items: List[HebrewItem]) -> str:
    return ''.join(str(item) for item in items).gsub(RAFE, '')



def iterate_dotted_text(text) #-> Iterator[HebrewItem]:

    n = text.length
    text += '  '
    i = 0
    while i < n:
        letter = text[i]

        dagesh = if can_dagesh(letter) then RAFE else '' end
        sin = if can_sin(letter) then RAFE else '' end
        niqqud = if can_niqqud(letter) then RAFE else '' end
        normalized = normalize(letter)
        i += 1

        nbrd = text[(i - 15)..(i + 15)].split()[1..-2] # check -1?

        # do we need something like that in ruby?
        # assert letter not in ANY_NIQQUD, f'{i}, {nbrd}, {[name_of(c) for word in nbrd for c in word]}'

        if is_hebrew_letter(normalized)
            if text[i] == DAGESH_LETTER
                dagesh = text[i]
                i += 1
            end
            if text[i] in NIQQUD_SIN
                sin = text[i]
                i += 1
            end
            if text[i] in NIQQUD
                niqqud = text[i]
                i += 1
            end
            if letter == 'ו' and dagesh == DAGESH_LETTER and niqqud == RAFE
                dagesh = RAFE
                niqqud = DAGESH_LETTER
            end
        end
        return HebrewChar(letter, normalized, dagesh, sin, niqqud)
      end
