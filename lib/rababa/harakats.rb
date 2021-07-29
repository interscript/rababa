# refers to:
# https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/util.py
# needing:
# https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py

require_relative "arabic_constants"

module Rababa::Harakats

    # Given stack, we extract its content to string, and check whether this string is
    # available at all_possible_haraqat list: if not we raise an error. When correct_reversed
    # is set, we also check the reversed order of the string, if it was not already correct.
    def extract_stack(stack, correct_reversed)
        char_haraqat = []

        while stack.length != 0
            char_haraqat.append(stack.pop)
        end

        full_haraqah = char_haraqat.join("")
        reversed_full_haraqah = char_haraqat.reverse().join("")

        if ArabicConstants::ALL_POSSIBLE_HARAQAT.include? full_haraqah
            out = full_haraqah
        elsif ArabicConstants::ALL_POSSIBLE_HARAQAT.include? reversed_full_haraqah &&  correct_reversed
            out = reversed_full_haraqah
        else
            val = full_haraqah.map{|diac| \
                    ArabicConstants::ALL_POSSIBLE_HARAQAT[diac]}.join('|')

            raise ValueError.new('The chart has the following haraqat which are
                                  not found in all possible haraqat: ' + val)
        end

        out
    end

    # Args:
    #     text (str): text to be diacritized
    # Returns:
    #     text: the text as came
    #     text_list: all text that are not haraqat
    #     vec_haraqat: all vec_haraqat
    def extract_haraqat(text, correct_reversed)
        if text.strip.length == 0
            return text, [" "] * text.length, [""] * text.length
        end

        stack = []
        vec_haraqat = []
        vec_txt = []
        text.chars.each do |char|
            # if chart is a diacritic, then extract the stack and empty it
            if !ArabicConstants::BASIC_HARAQAT.keys.include? char
                stack_content = extract_stack(stack, correct_reversed)
                vec_haraqat.push(stack_content)
                vec_txt.push(char)
                stack = []
            else
                stack.push(char)
            end
        end

        if vec_haraqat.length > 0
            vec_haraqat.shift
        end

        vec_haraqat.push(extract_stack(stack, true))

        [text, vec_txt, vec_haraqat]
    end

    # Args:
    #     text (str): text to be diacritized
    # Returns:
    #     text: the text as came
    #     #? text_list: all text that are not haraqat
    #     #? haraqat_list: all haraqat_list
    def remove_diacritics(text)
        Rababa::ArabicConstants::UBASIC_HARAQAT.keys.each do |diacritic|
            text.gsub(diacritic, "")
        end

        text
    end

    # 'a   a  a a'-> 'a a a a'
    def collapse_whitespace(text)
        text.gsub(/[[:space:]]+/, ' ')
    end

    # strip + remove redundancy in whitespaces
    def basic_cleaners(text)
        collapse_whitespace(text).strip
    end

    # filter arabic only + basic cleaner
    def valid_arabic_cleaners(text)
        text = text.chars.select {|c| Rababa::ArabicConstants::VALID_ARABIC.include? c}
        text = collapse_whitespace(text.join('')).strip
        text.strip
    end

end
