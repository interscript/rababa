"""
    run for now:
      ruby kithara.rb -t 'قطر' -m '../models-data/toy_model.onnx'
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

config_path = "../python/config/cbhg.yml" 
onnx_model_path = "../models-data/toy_model.onnx" # parser[:model]

diacritizer = Diacritizer::Diacritizer.new(onnx_model_path, config_path)
txt = diacritizer.diacritize_text(parser[:text])

output = onnx_processing("../models-data/toy_model.onnx")
