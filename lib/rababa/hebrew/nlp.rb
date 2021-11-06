# This file is completely inspired from
# https://github.com/elazarg/nakdimon/blob/master/hebrew.py

require 'rababa/hebrew/constants'

module Rababa
  module Hebrew
    module NLP
      class HebrewChar
        attr_accessor :letter, :normalized, :dagesh, :sin, :niqqud

        def initialize(letter, normalized, dagesh, sin, niqqud)
          @letter = letter
          @normalized = normalized
          @dagesh = dagesh
          @sin = sin
          @niqqud = niqqud
        end

        def to_str()
          self.letter + self.dagesh + self.sin + self.niqqud
        end

        def vocalize_dagesh(normalized, dagesh)
            if ! 'בכפ'.include? normalized # letter
                return ''
            end
            return dagesh.gsub(HebrewCONST::RAFE, '')
        end

        def vocalize_niqqud(c)
          # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
          if [Hebrew::Constants::Niqqud['KAMATZ'], Hebrew::Constants::Niqqud['PATAKH'],
              Hebrew::Constants::Niqqud['REDUCED_PATAKH']].include? c
            return Niqqud['PATAKH']
          end
          if [Hebrew::Constants::Niqqud['HOLAM'],
              Hebrew::Constants::Niqqud['REDUCED_KAMATZ']].include? c
            return Hebrew::Constants::Niqqud['HOLAM']
          end
          if [Hebrew::Constants::Niqqud['SHURUK'],
              Hebrew::Constants::Niqqud['KUBUTZ']].include? c
            return Hebrew::Constants::Niqqud['KUBUTZ']
          end
          if [Hebrew::Constants::Niqqud['TZEIRE'], Hebrew::Constants::Niqqud['SEGOL'],
              Hebrew::Constants::Niqqud['REDUCED_SEGOL']].include? c
            return Hebrew::Constants::Niqqud['SEGOL']
          end
          if c == Hebrew::Constants::Niqqud['SHVA']
            return ''
          end
          return c.gsub(Hebrew::Constants::RAFE, '')
        end

        def vocalize()
          niqqud = vocalize_niqqud(@niqqud)
          sin = @sin.gsub(Hebrew::Constants::RAFE, '')
          dagesh = vocalize_dagesh(@normalized, @dagesh)
          HebrewChar.new(@letter, @normalized, sin, dagesh, niqqud)
        end

      end # HebrewChar

      def numeric?(s)
        Float(s) != nil rescue false
      end

      def normalize(c)
        if Hebrew::Constants::VALID_LETTERS.include? c
          return c
        end
        if Hebrew::Constants::ENDINGS_TO_REGULAR.include? c
          return Hebrew::Constants::ENDINGS_TO_REGULAR[c]
        end
        if ['\n', '\t'].include? c
          return ' '
        end
        if ['־', '‒', '–', '—', '―', '−'].include? c
          return '-'
        end
        if c == '['
          return '('
        end
        if c == ']'
          return ')'
        end
        if ['´', '‘', '’'].include? c
          return "'"
        end
        if ['“', '”', '״'].include? c
          return '"'
        end
        if numeric?(c)
          return '5'
        end
        if c == '…'
          return ','
        end
        if ['ײ', 'װ', 'ױ'].include? c
          return 'H'
        end
        return 'O'
      end

      def is_hebrew_letter(letter)
        return ('\u05d0' <= letter) && (letter <= '\u05ea')
      end

      def can_dagesh(letter)
        return ('בגדהוזטיכלמנספצקשת' + 'ךף').include? letter
      end

      def can_sin(letter)
        return letter == 'ש'
      end

      def can_niqqud(letter)
        return ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן').include? letter
      end

      def can_any(letter)
        return can_niqqud(letter) || can_dagesh(letter) || can_sin(letter)
      end

    end
  end
end
