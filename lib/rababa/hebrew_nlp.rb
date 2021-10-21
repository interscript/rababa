# This file is completely inspired from
# https://github.com/elazarg/nakdimon/blob/master/hebrew.py

module Rababa

  module HebrewCONST

    RAFE = '\u05BF'
    NIQQUD_MAP = {
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

    HEBREW_LETTERS = %w[א ב ג ד ה ו ז ח ט י ך כ ל ם מ ן נ ס ע ף פ ץ צ ק ר ש ת]
    #(0x05d0..0x05ea + 1).to_a {|c| c.force_encoding('utf-8')}

    NIQQUD = [RAFE] + #HEBREW_LETTERS + \
        %w[ְ ֱ ֲ ֳ ִ ֵ ֶ ַ ָ ֹ ֺ ֻ ּ ַ]
    #        (0x05b0..0x05bc + 1).to_a {|c| c.force_encoding('utf-8')}

    HOLAM = NIQQUD_MAP['HOLAM']

    SHIN_YEMANIT = 'ׁ'
    #'\u05c1'
    SHIN_SMALIT = 'ׂ'
    #'\u05c2'
    NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]  # RAFE is for acronyms

    DAGESH_LETTER = 'ּ'
    #'\u05bc'
    DAGESH = [RAFE, DAGESH_LETTER]  # DAGESH and SHURUK are one and same


    ANY_NIQQUD = [RAFE] + NIQQUD[1..-1] + NIQQUD_SIN[1..-1] + DAGESH[1..-1]

    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + \
                HEBREW_LETTERS
    SPECIAL_TOKENS = %w[H O 5]

    ENDINGS_TO_REGULAR = Hash[
                  *('כמנפצ'.chars.zip 'ךםןףץ'.chars).map {|x,y| [x,y]}.flatten]

  end # HebrewCONST

  module HebrewNLP

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
        self.letter + self.dagesh + self.sin + self.niqqud
      end

      def vocalize_dagesh(normalized, dagesh)
        unless 'בכפ'.include? normalized # letter
          return ''
        end
        dagesh.gsub(HebrewCONST::RAFE, '')
      end

      def vocalize_niqqud(c)
        # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
        if [HebrewCONST::NIQQUD_MAP['KAMATZ'], HebrewCONST::NIQQUD_MAP['PATAKH'],
            HebrewCONST::NIQQUD_MAP['REDUCED_PATAKH']].include? c
          return NIQQUD_MAP['PATAKH']
        end
        if [HebrewCONST::NIQQUD_MAP['HOLAM'],
            HebrewCONST::NIQQUD_MAP['REDUCED_KAMATZ']].include? c
          return HebrewCONST::NIQQUD_MAP['HOLAM']
        end
        if [HebrewCONST::NIQQUD_MAP['SHURUK'],
            HebrewCONST::NIQQUD_MAP['KUBUTZ']].include? c
          return HebrewCONST::NIQQUD_MAP['KUBUTZ']
        end
        if [HebrewCONST::NIQQUD_MAP['TZEIRE'], HebrewCONST::NIQQUD_MAP['SEGOL'],
            HebrewCONST::NIQQUD_MAP['REDUCED_SEGOL']].include? c
          return HebrewCONST::NIQQUD_MAP['SEGOL']
        end
        if c == HebrewCONST::NIQQUD_MAP['SHVA']
          return ''
        end
        c.gsub(HebrewCONST::RAFE, '')
      end

      def vocalize
        niqqud = vocalize_niqqud(@niqqud)
        sin = @sin.gsub(HebrewCONST::RAFE, '')
        dagesh = vocalize_dagesh(@normalized, @dagesh)
        HebrewChar.new(@letter, @normalized, sin, dagesh, niqqud)
      end

    end # HebrewChar

    def numeric?(s)
      Float(s) != nil rescue false
    end

    def normalize(c)
      if HebrewCONST::VALID_LETTERS.include? c
        return c
      end
      if HebrewCONST::ENDINGS_TO_REGULAR.include? c
        return HebrewCONST::ENDINGS_TO_REGULAR[c]
      end
      if %w[\n \t].include? c
        return ' '
      end
      if %w[־ ‒ – — ― −].include? c
        return '-'
      end
      if c == '['
        return '('
      end
      if c == ']'
        return ')'
      end
      if %w[´ ‘ ’].include? c
        return "'"
      end
      if %w[“ ” ״].include? c
        return '"'
      end
      if numeric?(c)
        return '5'
      end
      if c == '…'
        return ','
      end
      if %w[ײ װ ױ].include? c
        return 'H'
      end
      'O'
    end

    def is_hebrew_letter(letter)
      ('\u05d0' <= letter) && (letter <= '\u05ea')
    end

    def can_dagesh(letter)
      ('בגדהוזטיכלמנספצקשת' + 'ךף').include? letter
    end

    def can_sin(letter)
      letter == 'ש'
    end

    def can_niqqud(letter)
      ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן').include? letter
    end

    def can_any(letter)
      can_niqqud(letter) || can_dagesh(letter) || can_sin(letter)
    end

  end # HebrewNLP

end
