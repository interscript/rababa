"""
this refers to:
  https://github.com/interscript/arabic-diacritization/blob/master/python/diacritizer.py
as well as half of
 https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py
"""

require 'onnxruntime'
require 'yaml'

require_relative "encoders"
require_relative "harakats"


module Diacritizer

    class Diacritizer

        # include config_manager
        # include text_encoder
        def initialize(onnx_model_path, config_path)

            # load inference model from model_path
            @onnx_session = OnnxRuntime::InferenceSession.new(onnx_model_path)
            @config = YAML.load_file(config_path)
            @encoder = get_text_encoder()
            p(@config)
            @vec_length = 197 #@config['max_len']
            # @config_manager = Config_manager::ConfigManager(config_path=config_path, model_kind=model_kind)
            # @config = @config_manager.config
            # @text_encoder = self.config_manager.text_encoder

            """ Required:
                * support 2 kind of encodings (encodings and decodings):
                    ArabicEncoderWithStartSymbol and BasicArabicEncoder
                * ? cpu + gpu & onnx?
            """

            @start_symbol_id = @encoder.start_symbol_id

        end

        def diacritize_text(text)

            # remove diacritics
            p(text)
            text = remove_diacritics(text)
            p(text)

            seq = @encoder.input_to_sequence(text)
            seq = seq+[0]*(@vec_length-seq.length)

            seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            batch_data = {'original': text,
                          'src': [seq],
                          'lengths': [seq.length]}

            text_out = diacritize_batch(batch_data)

            return text # mocked for now...
        end

        def diacritize_text(text):
            # convert string into indices
            seqs = self.text_encoder.input_to_sequence(text)
            # transform indices into "batch data"
            batch_data = {'original': text,
                          'src': [seqs],
                          'lengths': [len(seqs)]}

            return diacritize_batch(batch_data) #[0]

        def diacritize_file(path)
            """download data from relative path and diacritize it batch by batch"""
            data_iterator = get_data_from_file(path)
            diacritized_data = []

            data_iterator.each do |batch_inputs|
                batch_inputs["src"] = batch_inputs["src"].to(self.device)
                batch_inputs["lengths"] = batch_inputs["lengths"].to('cpu')
                batch_inputs["target"] = batch_inputs["target"].to(self.device)

                diacritize_batch(batch_inputs).each do |d|
                    diacritized_data.append(d)
                end
            end

            return diacritized_data
        end

        def get_data_from_file(path)
            """get data from relative path"""
            # data processed or not, specs in config file
            # data = IO.readlines(path, sep=@config_manager.config["data_separator"]) # [, open_args])
            # data iterator is just a list
            # batch_size = @config_manager.config["batch_size"]
            # data_iterator = array.each_slice(batch_size).to_a

            return data_iterator
        end

        def diacritize_batch(batch_data)

            # Call onnx model
            p(@onnx_session.inputs)

            # mocked
            ort_inputs = {@onnx_session.inputs[0][:name] => seqs}
            predictions = @onnx_session.run(nil, ort_inputs)

            """ real
            ort_inputs = {'src' => batch_data['src'],
                          'lengths' => batch_data['lengths']}
            predictions = @onnx_session.run(nil, ort_inputs)
            """
            sentences = []
            for i in (0.predictions.length()).to_a
                # combine cleaned arabic and predicted diacritics
                sentence = combine_text_and_haraqat(seqs[i], predictions[i])
                sentences.push(sentence)
            end

            return sentences
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
