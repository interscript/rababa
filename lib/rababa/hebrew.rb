require "rababa/hebrew/nlp"
require "rababa/hebrew/diacritizer"

module Rababa
  module Hebrew
    RAFE = '\u05BF'
    NIQQUD_HASH = {
      "SHVA" => '\u05B0',
      "REDUCED_SEGOL" => '\u05B1',
      "REDUCED_PATAKH" => '\u05B2',
      "REDUCED_KAMATZ" => '\u05B3',
      "HIRIK" => '\u05B4',
      "TZEIRE" => '\u05B5',
      "SEGOL" => '\u05B6',
      "PATAKH" => '\u05B7',
      "KAMATZ" => '\u05B8',
      "HOLAM" => '\u05B9',
      "KUBUTZ" => '\u05BB',
      "SHURUK" => '\u05BC',
      "METEG" => '\u05BD'
    }

    module Niqqud
      def self.[](name)
        NIQQUD_HASH[name]
      end
    end

    HEBREW_LETTERS = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "ך", "כ", "ל", "ם", "מ", "ן", "נ", "ס", "ע", "ף", "פ", "ץ", "צ", "ק", "ר", "ש", "ת"]
    # (0x05d0..0x05ea + 1).to_a {|c| c.force_encoding('utf-8')}

    NIQQUD = [RAFE] + # HEBREW_LETTERS + \
      ["ְ", "ֱ", "ֲ", "ֳ", "ִ", "ֵ", "ֶ", "ַ", "ָ", "ֹ", "ֺ", "ֻ", "ּ", "ַ"]
    #  (0x05b0..0x05bc + 1).to_a {|c| c.force_encoding('utf-8')}

    HOLAM = Niqqud["HOLAM"]

    SHIN_YEMANIT = "ׁ"
    # '\u05c1'
    SHIN_SMALIT = "ׂ"
    # '\u05c2'
    NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT] # RAFE is for acronyms

    DAGESH_LETTER = "ּ"
    # '\u05bc'
    DAGESH = [RAFE, DAGESH_LETTER] # DAGESH and SHURUK are one and same

    # rubocop:disable Style/SlicingWithRange
    ANY_NIQQUD = [RAFE] + NIQQUD[1..-1] + NIQQUD_SIN[1..-1] + DAGESH[1..-1]
    # rubocop:enable Style/SlicingWithRange

    VALID_LETTERS = [" ", "!", '"', "'", "(", ")", ",", "-", ".", ":", ";", "?"] + \
      HEBREW_LETTERS
    SPECIAL_TOKENS = ["H", "O", "5"]

    ENDINGS_TO_REGULAR = ("כמנפצ".chars.zip "ךםןףץ".chars).map { |x, y| [x, y] }.to_h
  end
end
