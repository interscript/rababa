"""
corresponds to:
https://github.com/interscript/arabic-diacritization/blob/master/python/util/text_encoders.py
and
https://github.com/interscript/arabic-diacritization/blob/master/python/util/text_cleaners.py
"""

require_relative "constant_arabic"
require_relative "encoders"


module Encoders

    class TextEncoder

        attr_accessor :start_symbol_id, :input_pad_id, \
                      :input_id_to_symbol, :target_id_to_symbol, \
                      :utarget_id_to_symbol

        def initialize(input_chars, target_charts,
                       cleaner,
                       reverse_input)

            # cleaner fcts
            @cleaner = cleaner

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
            @utarget_id_to_symbol = Hash[ \
                *Arabic_constant::UALL_POSSIBLE_HARAQAT.keys().map.with_index \
                                                      {|s, i| [i, s] }.flatten]

            @reverse_input = reverse_input
            @input_pad_id = @input_symbol_to_id[@pad]
            @start_symbol_id = nil

        end

        def clean(text)
            """cleaner, should be a method instantiated at init."""
            if @cleaner == "basic_cleaners"
                text = Harakats::basic_cleaners(text)
            elsif @cleaner == "valid_arabic_cleaners"
                text = Harakats::valid_arabic_cleaners(text)
            end
            return text
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

        def initialize(cleaner="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars()
            target_charts = Arabic_constant::ALL_POSSIBLE_HARAQAT.keys()

            super(input_chars, target_charts,
                  cleaner=cleaner,
                  reverse_input=reverse_input)
        end
    end

    class ArabicEncoderWithStartSymbol < TextEncoder

        def initialize(cleaner="basic_cleaners",
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars()
            target_charts = Arabic_constant::ALL_POSSIBLE_HARAQAT.keys()

            super(input_chars, target_charts,
                  cleaner=cleaner,
                  reverse_input=reverse_input)

            @start_symbol_id = @target_symbol_to_id["s"]

        end
    end
end
