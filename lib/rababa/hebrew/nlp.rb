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
          dagesh.gsub(HebrewCONST::RAFE, "")
        end

        def vocalize_niqqud(c)
          # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
          if [Niqqud["KAMATZ"], Niqqud["PATAKH"],
            Niqqud["REDUCED_PATAKH"]].include? c
            return Niqqud["PATAKH"]
          end
          if [Niqqud["HOLAM"],
            Niqqud["REDUCED_KAMATZ"]].include? c
            return Niqqud["HOLAM"]
          end
          if [Niqqud["SHURUK"],
            Niqqud["KUBUTZ"]].include? c
            return Niqqud["KUBUTZ"]
          end
          if [Niqqud["TZEIRE"], Niqqud["SEGOL"],
            Niqqud["REDUCED_SEGOL"]].include? c
            return Niqqud["SEGOL"]
          end
          if c == Niqqud["SHVA"]
            return ""
          end
          c.gsub(RAFE, "")
        end

        def vocalize
          niqqud = vocalize_niqqud(@niqqud)
          sin = @sin.gsub(RAFE, "")
          dagesh = vocalize_dagesh(@normalized, @dagesh)
          HebrewChar.new(@letter, @normalized, sin, dagesh, niqqud)
        end
      end # HebrewChar

      def numeric?(s)
        !Float(s).nil?
      rescue
        false
      end

      def normalize(c)
        if VALID_LETTERS.include? c
          return c
        end
        if ENDINGS_TO_REGULAR.include? c
          return ENDINGS_TO_REGULAR[c]
        end
        if ['\n', '\t'].include? c
          return " "
        end
        if ["־", "‒", "–", "—", "―", "−"].include? c
          return "-"
        end
        if c == "["
          return "("
        end
        if c == "]"
          return ")"
        end
        if ["´", "‘", "’"].include? c
          return "'"
        end
        if ["“", "”", "״"].include? c
          return '"'
        end
        if numeric?(c)
          return "5"
        end
        if c == "…"
          return ","
        end
        if ["ײ", "װ", "ױ"].include? c
          return "H"
        end
        "O"
      end

      def is_hebrew_letter(letter)
        ("\u05d0".."\u05ea").cover? letter
      end

      def can_dagesh(letter)
        ("בגדהוזטיכלמנספצקשת" + "ךף").include? letter
      end

      def can_sin(letter)
        letter == "ש"
      end

      def can_niqqud(letter)
        ("אבגדהוזחטיכלמנסעפצקרשת" + "ךן").include? letter
      end

      def can_any(letter)
        can_niqqud(letter) || can_dagesh(letter) || can_sin(letter)
      end
    end
  end
end
