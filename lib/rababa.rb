"""
  run text of file diacritization:
    ruby rababa.rb -t 'قطر' -m '../models-data/diacritization_model.onnx'
    ruby rababa.rb -f example.csv -m '../models-data/diacritization_model.onnx'

"""

require 'onnxruntime'
require 'optparse'
require 'yaml'

# load rababa library
require_relative "rababa/diacritizer"


def parser
  options = {}
  required_args = [:text, :model_path]
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
  [:model_path].each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil? }
  # p(options)
  options

end


parser = parser()

config_path = parser.has_key?(:config) ? parser[:config] : "config/model.yml"

diacritizer = Diacritizer::Diacritizer.new(parser[:model_path], config_path)

if parser.has_key?(:text)
    # run diacritization text if has :text
    txt = diacritizer.diacritize_text(parser[:text])
    p(txt)
elsif parser.has_key?(:text_filename)
    # run diacritization file
    txts = diacritizer.diacritize_file(parser[:text_filename])
    txts.each {|t| p(t)}
else
    raise ValueError.new('text or text_filename required')
end
