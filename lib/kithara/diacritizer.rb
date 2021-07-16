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
            @batch_size = @config['batch_size']

            # instantiate encoder's class
            @encoder = get_text_encoder()
            @start_symbol_id = @encoder.start_symbol_id

        end

        def preprocess_text(text)
            """preprocess text into indices"""
            #if (text.length > @max_length)
            #    raise ValueError.new('text length larger than max_length')
            #end

            text = @encoder.clean(text)
            text = Harakats::remove_diacritics(text)
            seq = @encoder.input_to_sequence(text)
            # correct expected length for vectors with 0's
            return seq+[0]*(@max_length-seq.length)
        end

        def diacritize_text(text)
            """Diacritize single arabic strings"""

            seq = preprocess_text(text)

            # initialize onnx computation
            # redondancy caused by batch processing of nnets
            ort_inputs = {'src' => [seq]*@batch_size,
                          'lengths' => [seq.length]*@batch_size}

            # onnx predictions
            preds = predict_batch(ort_inputs)[0]

            return combine_text_and_haraqat(seq, preds)
        end

        def diacritize_file(path)
            """download data from relative path and diacritize line by line"""

            texts = []
            File.open(path).each do |line|
                texts.push(line.chomp)
            end

            # process batches
            out_texts = []
            idx = 0
            loop do
                break if (idx+@batch_size > texts.length)

                src = texts[idx..idx+@batch_size-1].map.each{|t| \
                                                        preprocess_text(t)}
                lengths = src.map.each{|seq| seq.length}
                ort_inputs = {'src' => src,
                              'lengths' => lengths}
                preds = predict_batch(ort_inputs)

                out_texts += (0..@batch_size-1).map.each{|i| \
                                  combine_text_and_haraqat(src[i], preds[i])}
                idx += @batch_size
            end

            # process rest of data
            loop do
                break if (idx >= texts.length)
                out_texts += [diacritize_text(texts[idx])]
                idx += 1
            end

            return out_texts
        end

        def predict_batch(batch_data)
          # onnx predictions
          predicts = @onnx_session.run(nil, batch_data)
          predicts = predicts[0].map.each{|p| p.map.each{|r| r.each_with_index.max[1]}}
          return predicts
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

            return text.reverse
        end

        def get_text_encoder()
            """Initialise text encoder from config params"""
            if not ['basic_cleaners', 'valid_arabic_cleaners', nil].include? \
                                                @config['text_cleaner']
                raise Exception.new( \
                        'cleaner is not known: '+@config['text_cleaner'].to_s)
            end

            if @config['text_encoder'] == 'BasicArabicEncoder'
                encoder = Encoders::BasicArabicEncoder.new(@config['text_cleaner'])
            elsif @config['text_encoder'] == 'ArabicEncoderWithStartSymbol'
                encoder = Encoders::ArabicEncoderWithStartSymbol.new(@config['text_cleaner'])
            else
                raise Exception.new(\
                    'the text encoder is not found: '+@config['text_encoder'].to_s)
            end

            return encoder
        end

    end
end
