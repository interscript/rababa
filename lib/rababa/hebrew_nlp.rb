# This file is completely inspired from
# https://github.com/elazarg/nakdimon/blob/master/hebrew.py

module Rababa

  module HebrewCONST

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

    HEBREW_LETTERS = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף', 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת']
    #(0x05d0..0x05ea + 1).to_a {|c| c.force_encoding('utf-8')}

    NIQQUD = [RAFE] + #HEBREW_LETTERS + \
            ['ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ַ']
    #        (0x05b0..0x05bc + 1).to_a {|c| c.force_encoding('utf-8')}

    HOLAM = Niqqud['HOLAM']

    SHIN_YEMANIT = 'ׁ'
    #'\u05c1'
    SHIN_SMALIT = 'ׂ'
    #'\u05c2'
    NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]  # RAFE is for acronyms

    DAGESH_LETTER = 'ּ'
    #'\u05bc'
    DAGESH = [RAFE, DAGESH_LETTER]  # DAGESH and SHURUK are one and same

    ANY_NIQQUD = [RAFE] + NIQQUD[1..] + NIQQUD_SIN[1..] + DAGESH[1..]

    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + \
                HEBREW_LETTERS
    SPECIAL_TOKENS = ['H', 'O', '5']

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

      def to_str()
        self.letter + self.dagesh + self.sin + self.niqqud
      end

      def vocalize_dagesh(letter, dagesh)
          if ! 'בכפ'.include? letter
              return ''
          end
          return dagesh.gsub(HebrewCONST::RAFE, '')
      end

      def vocalize_niqqud(c)
        # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
        if [HebrewCONST::Niqqud['KAMATZ'], HebrewCONST::Niqqud['PATAKH'],
            HebrewCONST::Niqqud['REDUCED_PATAKH']].include? c
          return Niqqud['PATAKH']
        end
        if [HebrewCONST::Niqqud['HOLAM'],
            HebrewCONST::Niqqud['REDUCED_KAMATZ']].include? c
          return HebrewCONST::Niqqud['HOLAM']
        end
        if [HebrewCONST::Niqqud['SHURUK'],
            HebrewCONST::Niqqud['KUBUTZ']].include? c
          return HebrewCONST::Niqqud['KUBUTZ']
        end
        if [HebrewCONST::Niqqud['TZEIRE'], HebrewCONST::Niqqud['SEGOL'],
            HebrewCONST::Niqqud['REDUCED_SEGOL']].include? c
          return HebrewCONST::Niqqud['SEGOL']
        end
        if c == HebrewCONST::Niqqud['SHVA']
          return ''
        end
        return c.gsub(HebrewCONST::RAFE, '')
      end

      def vocalize()
        niqqud = vocalize_niqqud(@niqqud)
        sin = @sin.gsub(HebrewCONST::RAFE, '')
        dagesh = vocalize_dagesh(@letter, @dagesh)
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

  end # HebrewNLP

end
