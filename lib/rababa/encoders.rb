# corresponds to:
# https://github.com/interscript/rababa/blob/main/python/util/text_encoders.py
# and
# https://github.com/interscript/rababa/blob/main/python/util/text_cleaners.py

require_relative "hebrew_nlp"
require_relative "dataset"


module Encoders #Rababa

  #module Encoders

    class TextEncoder

      attr_accessor :normalized_table, :dagesh_table, :sin_table, :niqqud_table
      #include CharacterTable
      include Rababa::HebrewNLP

      def initialize()
        # cleaner fcts
        #@cleaner = get_text_cleaner()

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
      # def clean(text)
      #   text
      # end

      # Map string into Vector of HebrewChar
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
                    HebrewChar.new(letter, normalized, dagesh, sin, niqqud)
            )
          end
        end

        iterated__
      end

      # encode string into a Dataset::Data object
      def encode_data(text)
        # hebrew data
        data = encode_text(text)
        # Wrap data within Data structure representing language dims
        Dataset::Data.new(data.map.each {|d| d.letter},
                          data.map.each {|d|
                              @normalized_table.char_indices[d.normalized]},
                          data.map.each {|d|
                              @dagesh_table.char_indices[d.dagesh]},
                          data.map.each {|d|
                              @sin_table.char_indices[d.sin]},
                          data.map.each {|d|
                              @niqqud_table.char_indices[d.niqqud]})
      end

      # Combine initial original char and indices into a diacritised string
      # Args:
      #  text: char
      #  normalised: integer
      #  dagesh, dagesh, sin: integers
      # Returns:
      #   string
      def decode_idces(text, normalized, dagesh, sin, niqqud)
        Rababa::HebrewNLP::HebrewChar.new(text,
            @normalized_table.indices_char[normalized],
            @dagesh_table.indices_char[dagesh],
            @sin_table.indices_char[sin],
            @niqqud_table.indices_char[niqqud]).
                                      vocalize().to_str()
      end

      # Combine original and prediction vectors and return a string
      # Args:
      #   vtext: Array{str}, original text to be diacritized
      #   vnormalized: Array{integer}, hebrew indices
      #   vnormalized, vdagesh, vsin: Array{integer}, arrays of predicitions
      # Returns:
      #     text: the diacritized string
      def decode_data(vtext, vnormalized, vdagesh, vsin, vniqqud)
        dia_text = ''
        l_pred = vnormalized.length
        (0..l_pred-1).map.each {|i|
            dia_text +=
              decode_idces(vtext[i],vnormalized[i],vdagesh[i],vsin[i],vniqqud[i])
        }
        dia_text
      end

    end # TextEncoder


end # Encoders
