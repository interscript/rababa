

module Dataset

  class CharacterTable

    def initialise(chars)
      # make sure to be consistent with JS
      @MASK_TOKEN = ''
      @chars = [@MASK_TOKEN] + chars
      @char_indices = Hash[*@chars.map.with_index {|s, i| [s, i]}.flatten]
      @indices_char = Hash[*@chars.map.with_index {|s, i| [i, s]}.flatten]
    end

    def len()
      @chars.length
    end

    def to_ids(css)
      css.map.each {|cs| cs.map.each {|c| @char_indices[c]}}
    end

    def to_str(ids)
      ids.map.each {|cs| cs.map.each {|c| @indices_char[c]}}
    end

  end # CharacterTable


  class Data

    def initialise(text, normalized, dagesh, sin, niqqud)
      @text = text
      @normalized = normalized
      @dagesh = dagesh
      @sin = sin
      @niqqud = niqqud
    end

  end # Data

end
