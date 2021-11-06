module Rababa
  class Cleaner
    # strip + remove redundancy in whitespaces
    def clean(text)
      collapse_whitespace(text).strip
    end

    # 'a   a  a a'-> 'a a a a'
    def collapse_whitespace(text)
      text.gsub(/[[:space:]]+/, " ")
    end
  end
end
