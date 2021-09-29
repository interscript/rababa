# corresponds to:
# https://github.com/interscript/rababa/blob/main/python/util/text_encoders.py
# and
# https://github.com/interscript/rababa/blob/main/python/util/text_cleaners.py

require_relative "hebrew_nlp"
require_relative "cleaner"

module Rababa::Encoders

    class TextEncoder
        attr_accessor :start_symbol_id, :input_pad_id,
                      :input_id_to_symbol, :target_id_to_symbol,
                      :utarget_id_to_symbol

        def initialize(input_chars, target_chars,
                       cleaner_type,
                       reverse_input)

            # cleaner fcts
            @cleaner = get_text_cleaner(cleaner_type)

        end

        def get_text_cleaner(type)
          case type.to_s
          when 'basic_cleaners', nil
            Rababa::Cleaner::BasicCleaner.new
          when 'valid_arabic_cleaners'
            Rababa::Cleaner::ValidArabicCleaner.new
          else
            raise Exception.new(
              'text_cleaner not known: ' + type.to_s
            )
          end
        end

        # cleaner, instantiated at initialization
        def clean(text)
          @cleaner.clean(text)
        end

        # String -> Seq of chars -> List of indices
        def input_to_sequence(text)
            if @reverse_input
                text = text.chars.reverse.join("")
            end

            text.chars.map do |s|
                @input_symbol_to_id[s]
            end.map.reject{|i| i.nil?}
        end

    end

    class BasicArabicEncoder < TextEncoder

        def initialize(cleaner_type='basic_cleaners',
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            input_chars = "بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث".chars
            target_chars = Rababa::ArabicConstants::ALL_POSSIBLE_HARAQAT.keys

            super(
              input_chars,
              target_chars,
              cleaner_type=cleaner_type,
              reverse_input=reverse_input
            )
        end
    end

    class ArabicEncoderWithStartSymbol < BasicArabicEncoder

        def initialize(cleaner_type='basic_cleaners',
                       reverse_input: bool = false,
                       reverse_target: bool = false)

            super
            @start_symbol_id = @target_symbol_to_id["s"]
        end
    end
end
