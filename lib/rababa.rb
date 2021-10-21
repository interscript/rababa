# frozen_string_literal: true

# require_relative "rababa/arabic_constants"
require_relative "rababa/hebrew_nlp"
# require_relative "rababa/diacritizer"
# require_relative "rababa/encoders"
# require_relative "rababa/reconciler"
# require_relative "rababa/version"
require "optparse"
require "onnxruntime"
require "yaml"
require "tqdm"

print("jair")


module Rababa
  class Error < StandardError; end

  def self.parser
    options = {}

    OptionParser.new do |opts|
      opts.banner = "Usage: ruby_onnx.rb [options]"

      opts.on("-tTEXT", "--text=TEXT", "text to diacritize") do |t|
        options[:text] = t
      end
      opts.on("-fFILE", "--text_filename=FILE", "path to file to diacritize") do |f|
        options[:text_filename] = f
      end
      opts.on("-mMODEL", "--model_path=MODEL", "path to onnx model") do |m|
        options[:model_path] = m
      end
      opts.on("-cCONFIG", "--config=CONFIG", "path to config file") do |c|
        options[:config] = c
      end
    end.parse!
    # required args
    [:model_path].each do |arg|
      if options[arg].nil?
        raise OptionParser::MissingArgument, \
              arg
      end
    end
    # p(options)
    options

    print("jair")
  end
end
