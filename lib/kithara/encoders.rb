"""
corresponds to:
https://github.com/interscript/arabic-diacritization/blob/master/python/util/text_encoders.py
"""

require_relative "constant_arabic"
require_relative "encoders"


module Encoders

    class TextEncoder

        attr_accessor :start_symbol_id, :input_pad_id, \
                      :input_id_to_symbol, :target_id_to_symbol

        def initialize(input_chars, target_charts,
                       #cleaner_fn, #: Optional[str] = None
                       reverse_input) #: bool = false)

            #if cleaner_fn:
            #    self.cleaner_fn = getattr(text_cleaners, cleaner_fn)
            #else:
            #    self.cleaner_fn = None

            @pad = "P"
            @input_symbols = [@pad] + input_chars
            @target_symbols = [@pad] + target_charts

            # encoding of arabic without diacritics
            @input_symbol_to_id = Hash[*@input_symbols.map.with_index \
                                                  {|s, i| [s, i] }.flatten]
            @input_id_to_symbol = Hash[*@input_symbols.map.with_index \
                                                  {|s, i| [i, s] }.flatten]
            # encoding of haraqats
            @target_symbol_to_id = Hash[*@target_symbols.map.with_index \
                                                  {|s, i| [s, i] }.flatten]
            @target_id_to_symbol = Hash[*@target_symbols.map.with_index \
                                                  {|s, i| [i, s] }.flatten]

            @reverse_input = reverse_input
            @input_pad_id = @input_symbol_to_id[@pad]
            @start_symbol_id = nil

        end

        def input_to_sequence(text)
            """String -> Seq of chars -> List of indices """
            if @reverse_input
                text = text.chars().reverse.join("")
            end
            sequence = text.chars(). \
                            map{|s| @input_symbol_to_id[s]}. \
                                map.reject{|i| i.nil?}
            return sequence
        end

    end

    class BasicArabicEncoder < TextEncoder

        def initialize(#cleaner_fn="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars()
            target_charts = Arabic_constant::ALL_POSSIBLE_HARAQAT.keys()

            super(input_chars, target_charts,
                  #cleaner_fn=cleaner_fn,
                  reverse_input=reverse_input)
        end
    end

    class ArabicEncoderWithStartSymbol < TextEncoder

        def initialize(#cleaner_fn="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars()
            target_charts = Arabic_constant::ALL_POSSIBLE_HARAQAT.keys()

            super(input_chars, target_charts,
                  #cleaner_fn=cleaner_fn,
                  reverse_input=reverse_input)

            @start_symbol_id = @target_symbol_to_id["s"]

        end
    end
end
