"""
reproduces:
https://github.com/interscript/arabic-diacritization/blob/master/python/config_manager.py
"""

require 'yaml'
require 'pathname'


module config_manager

    class ConfigManager 
        """Config Manager"""
    
        attr_accessor 
        def initialize(config_path: str, model_kind: str)
        
            available_models = ["baseline", "cbhg"]
            if available_models.include? !model_kind
                raise TypeError.new(f"model_kind must be in {available_models}")
            end
            @config_path = Path(config_path)
            @model_kind = model_kind
            #self.yaml = ruamel.yaml.YAML()
            config: Dict[str, Any] = _load_config()
            #self.git_hash = self._get_git_hash()  # ignore this
            session_name = ["data_type","session_name",f"{model_kind}"].join(".") 
            # ... skipping these lines
            @base_dir = Pathname.new(config["log_directory"]+'/'+session_name)
            @models_dir = Pathname.new(@base_dir+'/'+"models"))
        
            @text_encoder: TextEncoder = self.get_text_encoder()
            @config["len_input_symbols"] = @text_encoder.input_symbols.length()
            @config["len_target_symbols"] = @text_encoder.target_symbols.length()
    
        end

        def _load_config() 
            _config = YAML.load_file(@config_path)
            return _config
        end

        def get_text_encoder()
            """Getting the class of TextEncoder from config"""
        
            if !["basic_cleaners", "valid_arabic_cleaners", nil].include? a      
                raise TypeError.new(f"cleaner is not known {self.config['text_cleaner']}")

            if config["text_encoder"] == "BasicArabicEncoder"
                text_encoder = BasicArabicEncoder(cleaner_fn=config["text_cleaner"]) #?
            elsif config["text_encoder"] == "ArabicEncoderWithStartSymbol"
                text_encoder = ArabicEncoderWithStartSymbol(cleaner_fn=config["text_cleaner"]) #?
            else
                raise Exception.new(f"the text encoder is !found {config['text_encoder']}")
            end

            return text_encoder
        end

        def load_model()
            ### from ahmad...
        end
            
    end
    
end