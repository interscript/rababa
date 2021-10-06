# corresponds to:
# https://github.com/interscript/rababa/blob/main/python/util/text_encoders.py
# and
# https://github.com/interscript/rababa/blob/main/python/util/text_cleaners.py

require_relative "hebrew_nlp"
# require_relative "cleaner"
require_relative "dataset"


module Encoders #Rababa

  #module Encoders

    class TextEncoder

      attr_accessor :normalized_table, :dagesh_table, :sin_table, :niqqud_table
      #include CharacterTable
      include Rababa::HebrewNLP

      def initialize()
        # cleaner fcts
        @cleaner = get_text_cleaner()
        # char tables
        @normalized_table = Dataset::CharacterTable.new(
                                    Rababa::HebrewCONST::SPECIAL_TOKENS + \
                                    Rababa::HebrewCONST::VALID_LETTERS)
        @dagesh_table = Dataset::CharacterTable.new(Rababa::HebrewCONST::DAGESH)
        @sin_table = Dataset::CharacterTable.new(Rababa::HebrewCONST::NIQQUD_SIN)
        @niqqud_table = Dataset::CharacterTable.new(Rababa::HebrewCONST::NIQQUD)
      end

      # def get_text_cleaner()
      # end

      # cleaner, instantiated at initialization
      def clean(text)
        text
      end

      # String -> Vect. of HebrewChar
      def encode_text(text)

        n = text.length
        text += '  '
        iterated__ = [] # "aggregator"

        i = 0
        while i < n
          letter = text[i]

          dagesh = if can_dagesh(letter) then Rababa::HebrewCONST::RAFE else '' end
          sin = if can_sin(letter) then Rababa::HebrewCONST::RAFE else '' end
          niqqud = if can_niqqud(letter) then Rababa::HebrewCONST::RAFE else '' end
          normalized = normalize(letter)

          i += 1

          # assert letter not in ANY_NIQQUD,
          # f'{i}, {nbrd}, {[name_of(c) for word in nbrd for c in word]}'

          if is_hebrew_letter(normalized)
            if letter == Rababa::HebrewCONST::DAGESH_LETTER
              dagesh = letter
              i += 1
            end
            if Rababa::HebrewCONST::NIQQUD_SIN.include? letter
              sin = letter
              i += 1
            end
            if Rababa::HebrewCONST::NIQQUD.include? letter
              niqqud = letter
              i += 1
            end
            if letter == 'ו' && \
                    dagesh == Rababa::HebrewCONST::DAGESH_LETTER && \
                    niqqud == Rababa::HebrewCONST::RAFE
              dagesh = RAFE
              niqqud = DAGESH_LETTER
            end
          end

          if normalized != 'O'
            iterated__.append(
                    HebrewChar.new(letter, normalized, dagesh, sin, niqqud))
          end
        end

        iterated__
      end

      def decode_idces(text, normalized, dagesh, sin, niqqud)
        Rababa::HebrewNLP::HebrewChar.new(text,
            @normalized_table.indices_char[normalized],
            @dagesh_table.indices_char[dagesh],
            @sin_table.indices_char[sin,
            @niqqud_table.indices_char[niqqud]).vocalize().to_str()
      end

      def decode_data(vtext, vnormalized, vdagesh, vsin, vniqqud)

        dia_text = ''
        l_pred = vnormalized.length
        (0..l_pred-1).map.each {|i|
            dia_text +=
            Rababa::HebrewNLP::HebrewChar.new(vtext[i],
                @encoder.normalized_table.indices_char[vnormalized[i]],
                @encoder.dagesh_table.indices_char[vdagesh[i]],
                @encoder.sin_table.indices_char[vsin[i]],
                @encoder.niqqud_table.indices_char[vniqqud[i]]).
                                                            vocalize().
                                                            to_str()
        }
        dia_text
        
      end

    end # TextEncoder

  # end

end # Encoders
