"""
this refers to:
  https://github.com/interscript/arabic-diacritization/blob/master/python/diacritizer.py
as well as half of
 https://github.com/almodhfer/Arabic_Diacritization/blob/master/config_manager.py
"""

from config_manager import ConfigManager
#from dataset import (DiacritizationDataset, collate_fn) # we might need collate
#from torch.utils.data import (DataLoader, 
#                              Dataset)

class Diacritizer:
    
    attr_accessor
    def initialize(config_path: str, model_kind: str, load_model: bool = false)
        
        @config_path = config_path
        @model_kind = model_kind
        @config_manager = ConfigManager(config_path=config_path, model_kind=model_kind)
        @config = @config_manager.config
        @text_encoder = self.config_manager.text_encoder
        
        """
        # !!! how to do this, it has to be a params?
        # @device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if load_model
            @model, @global_step = @config_manager.load_model()
            @model = @model.to(@device)
        end
        """

        @start_symbol_id = @text_encoder.start_symbol_id
        
    end
    
    def diacritize_text(text: str)
        # convert string into indices
        seq = @text_encoder.input_to_sequence(text)
        # transform indices into "batch data"
        batch_data = {'original' => text } #, 
        #              'src' => torch.Tensor([seq]).long(),
        #              'lengths' => torch.Tensor([len(seq)]).long()}
        
        return @diacritize_batch(batch_data)[0]
    end
    
    def diacritize_file(path: str)
        """download data from relative path and diacritize it batch by batch"""
        data_iterator = @get_data_from_file(path)
        diacritized_data = []
        data_iterator.each do |batch_inputs|  
            batch_inputs["src"] = batch_inputs["src"].to(self.device)
            batch_inputs["lengths"] = batch_inputs["lengths"].to('cpu')
            batch_inputs["target"] = batch_inputs["target"].to(self.device)
         
            @diacritize_batch(batch_inputs).each do |d|
                diacritized_data.append(d)
            end
        end

        return diacritized_data
    end
    
    def get_data_from_file(path)
            """get data from relative path"""
            loader_params = {"batch_size" => @config_manager.config["batch_size"],
                             "shuffle" => false, # do we need these 2?
                             "num_workers" => 2} # ???
            # data processed or not, specs in config file
            """ simpler and in ruby!
            if @config_manager.config["is_data_preprocessed"]
                data = pd.read_csv(path,
                                   encoding="utf-8",
                                   sep=self.config_manager.config["data_separator"],
                                   nrows=self.config_manager.config["n_validation_examples"],
                                   header=None)
            end

                # data = data[data[0] <= config_manager.config["max_len"]]
                dataset = DiacritizationDataset(self.config_manager, data.index, data)
            else:
                with open(path, encoding="utf8") as file:
                    data = file.readlines()
                data = [text for text in data if len(text) <= self.config_manager.config["max_len"]]
                dataset = DiacritizationDataset(self.config_manager, [idx for idx in range(len(data))], data)

            data_iterator = DataLoader(dataset, collate_fn=collate_fn, **loader_params)
            """
            # print(f"Length of data iterator = {len(valid_iterator)}")
            return data_iterator               
    end
    
end