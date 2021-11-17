# This file is completely inspired from
# https://github.com/elazarg/nakdimon/blob/master/hebrew.py

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

        def to_str
          letter + dagesh + sin + niqqud
        end

        def vocalize_dagesh(normalized, dagesh)
          unless "בכפ".include? normalized # letter
            return ""
          end
          dagesh.gsub(RAFE, "")
        end

        def vocalize_niqqud(c)
          # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
          case c
          when Niqqud::KAMATZ, Niqqud::PATAKH, Niqqud::REDUCED_PATAKH
            Niqqud::PATAKH
          when Niqqud::HOLAM, Niqqud::REDUCED_KAMATZ
            Niqqud::HOLAM
          when Niqqud::SHURUK, Niqqud::KUBUTZ
            Niqqud::KUBUTZ
          when Niqqud::TZEIRE, Niqqud::SEGOL, Niqqud::REDUCED_SEGOL
            Niqqud::SEGOL
          when Niqqud::SHVA
            ""
          else
            c.gsub(RAFE, "")
          end
        end

        def vocalize
          niqqud = vocalize_niqqud(@niqqud)
          sin = @sin.gsub(RAFE, "")
          dagesh = vocalize_dagesh(@normalized, @dagesh)
          HebrewChar.new(@letter, @normalized, sin, dagesh, niqqud)
        end
      end # HebrewChar

      def normalize(c)
        case c
        when *VALID_LETTERS
          c
        when *ENDINGS_TO_REGULAR.keys
          ENDINGS_TO_REGULAR[c]
        when "\n", "\t"
          " "
        when "־", "‒", "–", "—", "―", "−"
          "-"
        when "["
          "("
        when "]"
          ")"
        when "´", "‘", "’"
          "'"
        when "“", "”", "״"
          '"'
        when ("0".."9")
          "5"
        when "…"
          ","
        when "ײ", "װ", "ױ"
          "H"
        else
          "O"
        end
      end

      def is_hebrew_letter?(letter)
        ("\u05d0".."\u05ea").cover? letter
      end

      def can_dagesh?(letter)
        ("בגדהוזטיכלמנספצקשת" + "ךף").include? letter
      end

      def can_sin?(letter)
        letter == "ש"
      end

      def can_niqqud?(letter)
        ("אבגדהוזחטיכלמנסעפצקרשת" + "ךן").include? letter
      end

      def can_any?(letter)
        can_niqqud?(letter) || can_dagesh?(letter) || can_sin?(letter)
      end
    end
  end
end
