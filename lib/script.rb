# frozen_string_literal: true

# RUNS:
# ruby script.rb -m ../models-data/diacritization_model.onnx --t 'מה שלומך'
# ruby script.rb -m ../models-data/diacritization_model.onnx --f '../python/data/test/test.txt'

require_relative "rababa/hebrew_nlp"
require_relative "rababa/diacritizer"

require "optparse"
require "onnxruntime"
require "yaml"
require "tqdm"

def parser
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
  [:model_path].each { |arg| raise OptionParser::MissingArgument, arg if options[arg].nil? }
  # p(options)
  options
end

config_path = parser.key?(:config) ? parser[:config] : "../config/model.yml"

config = YAML.safe_load(File.read(config_path))

diacritizer = Rababa::Diacritizer.new(parser[:model_path], config)

if parser.key?(:text)
  # run diacritization text if has :text
  txt = diacritizer.diacritize_text(parser[:text])
  p(txt)
elsif parser.key?(:text_filename)
  # run diacritization file
  txts = diacritizer.diacritize_file(parser[:text_filename])
  txts.each { |t| p(t) }
else
  raise ValueError, "text or text_filename required"
end
