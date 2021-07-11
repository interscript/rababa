"""
this refers to:
  https://github.com/interscript/arabic-diacritization/blob/master/python/diacritizer.py
as well as half of
 https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py
"""

require 'onnxruntime'
require 'yaml'


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
        def initialize(model_path, config_path)
        
            # load inference model from model_path
            @onnx_session = OnnxRuntime::InferenceSession.new(model_path)
            @config = YAML.load_file(config_path)
            
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
            #data = IO.readlines(path, sep=@config_manager.config["data_separator"]) # [, open_args])
        
            # data iterator is just a list
            # batch_size = @config_manager.config["batch_size"]
            # data_iterator = array.each_slice(batch_size).to_a
        
            return data_iterator               
        end
        
        
        def diacritize_batch(seq)
            
            # Call onnx model
            ort_inputs = {@onnx_session.inputs[0][:name] => seq}
            out = @onnx_session.run(nil, ort_inputs)
            
            return out
        end
    
    end
    
end
