
module Rababa::Cleaner

  class BasicCleaner
    # strip + remove redundancy in whitespaces
    def clean(text)
      collapse_whitespace(text).strip
    end

    # 'a   a  a a'-> 'a a a a'
    def collapse_whitespace(text)
      text.gsub(/[[:space:]]+/, ' ')
    end
  end

  class ValidArabicCleaner < BasicCleaner
    # filter arabic only + basic cleaner
    def clean(text)
      text = text.chars.select {|c| Rababa::ArabicConstants::VALID_ARABIC.include? c}
      text = collapse_whitespace(text.join('')).strip
      text.strip
    end
  end

end
