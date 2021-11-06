# frozen_string_literal: true

require "rababa/version"
require "rababa/arabic"
require 'optparse'
require 'onnxruntime'
require 'yaml'
require 'tqdm'

module Rababa
  class Error < StandardError; end

  def self.parser
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
      opts.on("-lLANGUAGE", "--language=LANGUAGE", "select a language") do |l="arabic"|
        options[:language] = l
      end

    end.parse!
    # required args
    [:model_path].each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil? }
    # p(options)
    options

  end

end
