
# this refers to:
#   https://github.com/interscript/rababa/blob/main/python/diacritizer.py
# as well a drastic simplification of
#   https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py


require_relative 'encoders'
require_relative 'harakats'
require_relative 'reconcile'


module Rababa

    class Diacritizer
        include Rababa::Harakats
        include Rababa::Reconcile

        def initialize(onnx_model_path, config)

            # load inference model from model_path
            @onnx_session = OnnxRuntime::InferenceSession.new(onnx_model_path)

            # load config
            @config = config
            @max_length = @config['max_len']
            @batch_size = @config['batch_size']

            # instantiate encoder's class
            @encoder = get_text_encoder
            @start_symbol_id = @encoder.start_symbol_id

        end

        # preprocess text into indices
        def preprocess_text(text)
            # if (text.length > @max_length)
            #     raise ValueError.new('text length larger than max_length')
            # end
            # hack in absence of preprocessing!
            if text.length > @max_length
                text = text[0..@max_length]
                warn('WARNING:: string cut length > #{@max_length},\n')
                warn('text:: '+text)
            end

            text = @encoder.clean(text)
            text = remove_diacritics(text)
            seq = @encoder.input_to_sequence(text)
            # correct expected length for vectors with 0's
            return seq+[0]*(@max_length-seq.length)
        end

        # Diacritize single arabic strings
        def diacritize_text(text)
            """Diacritize single arabic strings"""
            text = text.strip()
            seq = preprocess_text(text)

            # initialize onnx computation
            # redondancy caused by batch processing of nnets
            ort_inputs = {
                'src' => [seq]*@batch_size,
                'lengths' => [seq.length]*@batch_size
            }

            # onnx predictions
            preds = predict_batch(ort_inputs)[0]

            reconcile_strings(text, combine_text_and_haraqat(seq, preds))
        end

        # download data from relative path and diacritize line by line
        def diacritize_file(path)
            texts = []
            File.open(path).each do |line|
                texts.push(line.chomp.strip())
            end

            # process batches
            out_texts = []
            idx = 0
            loop do
                break if (idx+@batch_size > texts.length)

                originals = texts[idx..idx+@batch_size-1]
                src = originals.map.each{|t| preprocess_text(t)}
                lengths = src.map.each{|seq| seq.length}
                ort_inputs = {'src' => src,
                              'lengths' => lengths}
                preds = predict_batch(ort_inputs)

                out_texts += (0..@batch_size-1).map.each{|i| \
                  reconcile_strings(originals[i],
                                    combine_text_and_haraqat(src[i], preds[i]))
                }
                idx += @batch_size
            end

            # process rest of data
            loop do
                break if (idx >= texts.length)
                out_texts += [diacritize_text(texts[idx])]
                idx += 1
            end

            out_texts
        end

        # Call ONNX model with data transformed in batches
        def predict_batch(batch_data)
          # onnx predictions
          predicts = @onnx_session.run(nil, batch_data)
          predicts = predicts[0].map.each{|p| \
                                    p.map.each{|r| r.each_with_index.max[1]}}
          return predicts
        end

        # Combine: text + Haraqats --> diacritised arabic
        def combine_text_and_haraqat(vec_txt, vec_haraqat, encoding_mode='std')
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

                if encoding_mode == 'std'
                    s = @encoder.input_id_to_symbol[txt].to_s + \
                            @encoder.target_id_to_symbol[haraq].to_s

                elsif encoding_mode == 'escaped unicode'
                    s = @encoder.input_id_to_symbol[txt].to_s + \
                            @utarget_symbol_to_id.utarget_id_to_symbol[haraq].to_s
                end
                text += s
            end

            text #.reverse
        end

        # Initialise text encoder from config params
        def get_text_encoder()
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

            encoder
        end

    end
end
