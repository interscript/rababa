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
      #include CharacterTable
      #attr_accessor
      def initialize()
        # cleaner fcts
        @cleaner = get_text_cleaner()
        # char tables
        #@letters_table = CharacterTable.new(Rababa::HebrewCONST::SPECIAL_TOKENS + \
        #                                    Rababa::HebrewCONST::VALID_LETTERS)
        #print(Rababa::HebrewCONST::DAGESH.class)
        #@dagesh_table = CharacterTable.new(Rababa::HebrewCONST::DAGESH)
        #@sin_table = CharacterTable.new(Rababa::HebrewCONST::NIQQUD_SIN)
        #@niqqud_table = CharacterTable.new(Rababa::HebrewCONST::NIQQUD)
      end

      def get_text_cleaner()
      end

      # cleaner, instantiated at initialization
      def clean(text)
        # @cleaner.clean(text)
        text
      end

      # String -> Vect. of HebrewChar
      def encode_dotted_text(str)

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

          nbrd = text[(i - 15)..(i + 15)].split()[1..-1]
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

          iterated__.append(
                HebrewNLP::HebrewChar(letter, normalized, dagesh, sin, niqqud))
        end

        iterated__
      end

    end # TextEncoder

  # end

end # Encoders
