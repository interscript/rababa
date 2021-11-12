require "rababa/cleaner"

module Rababa
  module Arabic
    class Cleaner < Rababa::Cleaner
      # filter arabic only + basic cleaner
      def clean(text)
        text = text.chars.select {|c| Rababa::Arabic::Constants::VALID_ARABIC.include? c}
        text = collapse_whitespace(text.join('')).strip
        text.strip
      end
    end
  end
end

