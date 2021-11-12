require "rababa"

module Rababa
  module CLI
    module_function

    def cli_options
      options = {}
      OptionParser.new do |opts|
        opts.banner = "Usage: #{$0} [options]"

        opts.on("-t", "--text=TEXT", String, "text to diacritize") do |t|
          options[:text] = t
        end
        opts.on("-f", "--text_filename=FILE", String, "path to file to diacritize") do |f|
          options[:text_filename] = f
        end
        opts.on("-m", "--model_path=MODEL", String, "path to ONNX model") do |m|
          options[:model_path] = m
        end
        opts.on("-c", "--config=CONFIG", String, "path to config file") do |c|
          options[:config] = c
        end
        opts.on("-l", "--language=LANGUAGE", String, "select a language (arabic, hebrew)") do |l|
          options[:language] = l
        end
      end.parse!
      options[:language] ||= "arabic"

      options
    end

    def call
      parser = cli_options

      config_path = parser[:config]
      model_path = parser[:model_path]

      case parser[:language]
      when "arabic"
        config_path ||= "config/model_arabic.yml"
        model_path ||= "models-data/diacritization_model_ARABIC.onnx"
        p(model_path)
        diacritizer_class = Rababa::Arabic::Diacritizer
      when "hebrew"
        config_path ||= "config/model_hebrew.yml"
        model_path ||= "models-data/diacritization_model_HEBREW.onnx"
        diacritizer_class = Rababa::Hebrew::Diacritizer
      else
        raise ArgumentError, "#{parser[:language]} is unsupported"
      end

      diacritizer = diacritizer_class.new(
        model_path,
        YAML.load_file(config_path)
      )

      if parser.key?(:text)
        # run diacritization text if has :text
        txt = diacritizer.diacritize_text(parser[:text])
        puts txt
      elsif parser.key?(:text_filename)
        # run diacritization file
        txts = diacritizer.diacritize_file(parser[:text_filename])
        txts.each { |t| puts t }
      else
        raise ValueError, "text or text_filename required"
      end
    end
  end
end
