# frozen_string_literal: true

# directly inspired from
# https://github.com/elazarg/nakdimon/blob/master/dataset.py

module Dataset
  class CharacterTable
    attr_accessor :char_indices, :indices_char

    def initialize(chars)
      # make sure to be consistent with JS
      @mask_token = ""
      @chars = [@mask_token] + chars
      @char_indices = Hash[*@chars.map.with_index { |s, i| [s, i] }.flatten]
      @indices_char = Hash[*@chars.map.with_index { |s, i| [i, s] }.flatten]
    end

    def len
      @chars.length
    end

    def to_ids(css)
      # css.map.each {|cs| cs.map.each {|c| @char_indices[c]}}
      css.map.each { |c| @char_indices[c] }
    end

    def to_str(ids)
      ids.map.each { |cs| cs.map.each { |c| @indices_char[c] } }
    end
  end

  class Data
    attr_accessor :text, :normalized, :dagesh, :sin, :niqqud

    def initialize(vtext, vnormalized, vdagesh, vsin, vniqqud)
      @text = vtext
      @normalized = vnormalized
      @dagesh = vdagesh
      @sin = vsin
      @niqqud = vniqqud
    end
  end
end
