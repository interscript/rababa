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
    opts.on("-mMODEL", "--model=MODEL", "path to onnx model") do |m|
      options[:model_path] = m
    end
  end.parse!
  required_args.each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil?  }
  options
end


parser = parser()

config_path = "configs/cbhg.yml"
# onnx_model_path = "../models-data/diacritization_model.onnx"
p(parser[:model])
diacritizer = Diacritizer::Diacritizer.new(parser[:model_path], config_path)

# run diacritize text
txt = diacritizer.diacritize_text(parser[:text])
p(txt)

# run diacritize file
txts = diacritizer.diacritize_file('example.csv')
# p(txts)
txts.each {|t| p(t)}
