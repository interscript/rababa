"""
this refers to:
  https://github.com/interscript/arabic-diacritization/blob/master/python/diacritizer.py
as well a drastic simplification of
 https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py
"""

require 'onnxruntime'
require 'yaml'
require 'tqdm'

require_relative "encoders"
require_relative "harakats"

include Harakats


module Diacritizer

    class Diacritizer

        def initialize(onnx_model_path, config_path)

            # load inference model from model_path
            @onnx_session = OnnxRuntime::InferenceSession.new(onnx_model_path)

            # load config
            @config = YAML.load_file(config_path)
            @max_length = @config['max_len']
            @batch_size = 32 # @config['batch_size']

            # instantiate encoder's class
            @encoder = get_text_encoder()
            @start_symbol_id = @encoder.start_symbol_id

            """ TODO:
                * cleaner fct?
            """

        end

        def diacritize_text(text)
            """Diacritize single arabic strings"""

            # remove diacritics
            text = Harakats::remove_diacritics(text)
            # map input to idces
            seq = @encoder.input_to_sequence(text)
            # correct expected lenght for vectors
            seq = seq+[0]*(@max_length-seq.length)

            # initialize onnx computation
            ort_inputs = {'src' => [seq]*@batch_size,
                          'lengths' => [seq.length]*@batch_size}
            # onnx predictions
            predicts = @onnx_session.run(nil, ort_inputs)
            # network outputs some likelihood for each haraqat:
            preds = predicts[0][1].map.each{|r| r.each_with_index.max[1]}

            # combine input sequence with predicted harakats
            return combine_text_and_haraqat(seq, preds)
        end

        def diacritize_file(path)
            """download data from relative path and diacritize line by line"""

            in_texts = []
            File.open(path).each do |line|
                in_texts.push(line.chomp)
            end

            return in_texts.tqdm.map {|t| diacritize_text(t)}
        end

        def combine_text_and_haraqat(vec_txt, vec_haraqat)

            if vec_txt.length != vec_haraqat.length
                raise Exception.new('haraqat.len != txt.len in \
                                     Harakats::combine_text_and_haraqat')
            end

            text, i = '', 0
            loop do
                txt = vec_txt[i]
                haraq = vec_haraqat[i]
                i += 1
                break if (i == vec_txt.length) or \
                          (txt == @encoder.input_pad_id)
                text += @encoder.input_id_to_symbol[txt] + \
                        @encoder.target_id_to_symbol[haraq]
            end

            return text
        end

        def get_text_encoder()
            """Initialise text encoder from config params"""
            if not ['basic_cleaners', 'valid_arabic_cleaners', nil].include? \
                                                @config['text_cleaner']
                raise Exception.new( \
                        'cleaner is not known: '+@config['text_cleaner'].to_s)
            end

            if @config['text_encoder'] == 'BasicArabicEncoder'
                encoder = Encoders::BasicArabicEncoder.new()
            elsif @config['text_encoder'] == 'ArabicEncoderWithStartSymbol'
                encoder = Encoders::ArabicEncoderWithStartSymbol.new()
            else
                raise Exception.new(\
                    'the text encoder is not found: '+@config['text_encoder'].to_s)
            end

            return encoder
        end

    end
end
