require "rababa/cleaner"

module Rababa
  module Arabic
    class Cleaner < Rababa::Cleaner
      # filter arabic only + basic cleaner
      def clean(text)
        text = text.chars.select { |c| VALID_ARABIC.include? c }.join
        text = super(text)
        text.strip
      end
    end
  end
end
