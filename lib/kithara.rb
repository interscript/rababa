"""
    run:
      ruby kithara.rb -t 'قطر' -m '../models-data/diacritization_model.onnx'
      ruby kithara.rb -f example.csv -m '../models-data/diacritization_model.onnx'

"""

require 'onnxruntime'
require 'optparse'
require 'yaml'

# load kithara library
require_relative "kithara/diacritizer"


def parser
  options = {}
  required_args = [:text, :model_path]
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby_onnx.rb [options]"

    opts.on("-tTEXT", "--text=TEXT", "text to diacritize") do |t|
      options[:text] = t
    end
    opts.on("-fFILE", "--file=FILE", "path to file to diacritize") do |f|
      options[:text_filename] = f
    end
    opts.on("-mMODEL", "--model=MODEL", "path to onnx model") do |m|
      options[:model_path] = m
    end
    opts.on("-cMODEL", "--config=MODEL", "path to config file") do |c|
      options[:model_path] = c
    end

  end.parse!
  # required args
  [:model_path].each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil?  }
  # p(options)
  options

end


parser = parser()

config_path = parser.has_key?(:config) ? parser[:config] : "config/model.yml"

diacritizer = Diacritizer::Diacritizer.new(parser[:model_path], config_path)

if parser.has_key?(:text)
    # run diacritize text if has :text
    txt = diacritizer.diacritize_text(parser[:text])
    p(txt)
elsif parser.has_key?(:text_filename)
    # run diacritize file
    txts = diacritizer.diacritize_file('example.csv')
    txts.each {|t| p(t)}
else
    raise ValueError.new('text or text_filename required')
end
