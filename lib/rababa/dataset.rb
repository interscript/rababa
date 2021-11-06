# directly inspired from
# https://github.com/elazarg/nakdimon/blob/master/dataset.py

module Dataset
  class CharacterTable
    attr_accessor :char_indices, :indices_char

    def initialize(chars)
      # make sure to be consistent with JS
      @mask_token = ""
      @chars = [@mask_token] + chars
      @char_indices = @chars.map.with_index { |s, i| [s, i] }.to_h
      @indices_char = @chars.map.with_index { |s, i| [i, s] }.to_h
    end

    def len
      @chars.length
    end

    def to_ids(css)
      # css.map {|cs| cs.map {|c| @char_indices[c]}}
      css.map { |c| @char_indices[c] }
    end

    def to_str(ids)
      ids.map { |cs| cs.map { |c| @indices_char[c] } }
    end
  end # CharacterTable

  class Data
    attr_accessor :text, :normalized, :dagesh, :sin, :niqqud

    def initialize(vtext, vnormalized, vdagesh, vsin, vniqqud)
      @text = vtext
      @normalized = vnormalized
      @dagesh = vdagesh
      @sin = vsin
      @niqqud = vniqqud
    end
  end # Data
end
