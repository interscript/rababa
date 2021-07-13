"""
this refers to:
  https://github.com/interscript/arabic-diacritization/blob/master/python/diacritizer.py
as well as half of
 https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py
"""

require 'onnxruntime'
require 'yaml'

require_relative "encoders"

#from config_manager import ConfigManager
#from dataset import (DiacritizationDataset, collate_fn) # we might need collate
#from torch.utils.data import (DataLoader,
#                              Dataset)

# require_relative "config_manager"
# require_relative "text_encoder"


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
            # @config_manager = Config_manager::ConfigManager(config_path=config_path, model_kind=model_kind)
            # @config = @config_manager.config
            # @text_encoder = self.config_manager.text_encoder

            """ Required:
                * support 2 kind of encodings (encodings and decodings):
                    ArabicEncoderWithStartSymbol and BasicArabicEncoder
                * ? cpu + gpu & onnx?
            """

            # @start_symbol_id = @text_encoder.start_symbol_id

        end

        def diacritize_text(text)

          p(text)
          # remove diacritics
          text = remove_diacritics(text)
          p(text)

          seq = @encoder.input_to_sequence(text2)
          
            # encoder: string -> hot encoding
            #     seq = @text_encoder.input_to_sequence(text)
            batch_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0]]

            text_out = diacritize_batch(batch_data) #[0]
            # p(text_out)

            return text # mocked for now...
        end

        def diacritize_text(self, text: str):
        # convert string into indices
        seq = self.text_encoder.input_to_sequence(text)
        # transform indices into "batch data"
        batch_data = {'original': text,
                      'src': torch.Tensor([seq]).long(),
                      'lengths': torch.Tensor([len(seq)]).long()}

        return self.diacritize_batch(batch_data)[0]

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


        def diacritize_batch(seq)

            # Call onnx model
            p(@onnx_session.inputs)

            ort_inputs = {@onnx_session.inputs[0][:name] => seq}
            out = @onnx_session.run(nil, ort_inputs)

            return out
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
