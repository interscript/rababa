"""
refers to:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/util.py
needing:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py
"""

require_relative "arabic_constant"

include arabic_constant


def extract_haraqat(text: str, correct_reversed: bool = true) 
    """
    Args:
    text (str): text to be diacritized
    Returns:
    text: the text as came
    text_list: all text that are !haraqat
    haraqat_list: all haraqat_list
    """
    if text.strip().length() == 0 
        return text, [" "] * text.length(), [""] * text.length()
    end
    stack = []
    haraqat_list = []
    txt_list = []
    text.each do |char| 
        # if chart is a diacritic, then extract the stack and empty it
        if !arabic_constant::BASIC_HARAQAT.keys().include? char  
            stack_content = extract_stack(stack, correct_reversed=correct_reversed)
            haraqat_list.push(stack_content)
            txt_list.push(char)
            stack = []
        else
            stack.push(char)
        end
    end
    if haraqat_list.length() > 0 
        del haraqat_list[0]
    end
    haraqat_list.push(extract_stack(stack))
    
    return text, txt_list, haraqat_list
end
    

def remove_diacritics(text: str)
    """
    Args:
        text (str): text to be diacritized
    Returns:
        text: the text as came
        #? text_list: all text that are not haraqat
        #? haraqat_list: all haraqat_list
    """
    for diacritic in arabic_constant::BASIC_HARAQAT.keys()
        text = text.replace(diacritic, "")
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
    
    assert txt_list.length() == haraqat_list.length()
    out = []
    in enumerate(txt_list).each do |i,char|  
        out.push(char)
        out.push(haraqat_list[i])
    end
    return  out.join("") 
end
