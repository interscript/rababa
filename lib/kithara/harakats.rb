"""
refers to:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/util.py
needing:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py
"""

#require_relative "arabic_constant"

require_relative "constant_arabic"

module Harakats

    def extract_stack(stack, correct_reversed)
        """
        Given stack, we extract its content to string, and check whether this string is
        available at all_possible_haraqat list: if not we raise an error. When correct_reversed
        is set, we also check the reversed order of the string, if it was not already correct.
        """
        char_haraqat = []
        while stack.length() != 0
            char_haraqat.append(stack.pop())
        end

        full_haraqah = char_haraqat.join("")
        reversed_full_haraqah = char_haraqat.reverse().join("")

        if Arabic_constant::ALL_POSSIBLE_HARAQAT.include? full_haraqah
            out = full_haraqah
        elsif Arabic_constant::ALL_POSSIBLE_HARAQAT.include? reversed_full_haraqah  &&  correct_reversed
            out = reversed_full_haraqah
        else
            val = full_haraqah.map{|diac| \
                    Arabic_constant::ALL_POSSIBLE_HARAQAT[diac]}.join('|')

            raise ValueError.new('The chart has the following haraqat which are
                                  not found in all possible haraqat: ' + val)
        end

        return out
    end


    def extract_haraqat(text, correct_reversed)
        """
        Args:
            text (str): text to be diacritized
        Returns:
            text: the text as came
            text_list: all text that are not haraqat
            haraqat_list: all haraqat_list
        """
        if text.strip().length() == 0
            return text, [" "] * text.length(), [""] * text.length()
        end

        stack = []
        haraqat_list = []
        txt_list = []
        text.chars().each do |char|
            # if chart is a diacritic, then extract the stack and empty it
            if !Arabic_constant::BASIC_HARAQAT.keys().include? char
                stack_content = extract_stack(stack, correct_reversed)
                haraqat_list.push(stack_content)
                txt_list.push(char)
                stack = []
            else
                stack.push(char)
            end
        end
        if haraqat_list.length() > 0
            haraqat_list.shift
        end
        haraqat_list.push(extract_stack(stack, true))

        return text, txt_list, haraqat_list
    end

    def remove_diacritics(text)
        """
        Args:
            text (str): text to be diacritized
        Returns:
            text: the text as came
            #? text_list: all text that are not haraqat
            #? haraqat_list: all haraqat_list
        """
        Arabic_constant::BASIC_HARAQAT.keys().each do |diacritic|
            text.gsub(diacritic, "")
        end

        return text
    end


    def combine_txt_and_haraqat(txt_list, haraqat_list)
        """
          Rejoins text with its corresponding haraqat
          Args:
              txt_list: The text that does !contain any haraqat
              haraqat_list: The haraqat that are corresponding to the text list
        """
        assert_equal(txt_list.length(), haraqat_list.length(), \
                 failure_message = "haraqat_list.len != txt_list.len")

        out = []
        for i in (0..txt_list.length).to_a
            out.push(txt_list[i])
            out.push(haraqat_list[i])
        end

        return  out.join("")
    end

end
