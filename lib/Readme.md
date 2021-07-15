# Arabic Diacritization in Ruby with Kithara

### Run examples
One can diacritize either single strings:

	ruby kithara.rb -t 'قطر' -m '../models-data/diacritization_model.onnx'
Or files:

	ruby kithara.rb -f 'data/example.csv' -m '../models-data/diacritization_model.onnx'

### Parameters
* text to diacritize: "**-t**TEXT", "--text=TEXT",
* path to file to diacritize: "**-f**FILE", "--text_filename=FILE",
* path to onnx model **Mandatory**: "-mMODEL", "--model_file=MODEL",
* path to config file **Default:config/model.yml**: "-cCONFIG", "--config=CONFIG"

### Config
#### Players:
* max_len: 600
	longer sentences will need to be preprocessed, which can be done for instance using [Hamza5](https://github.com/Hamza5) [code](https://github.com/Hamza5/Pipeline-diacritizer/blob/master/pipeline_diacritizer/pipeline_diacritizer.py).
* text_encoder:
     * BasicArabicEncoder
     *  ArabicEncoderWithStartSymbol
* text_cleaner:
     * basic_cleaners
     * valid_arabic_cleaners

### ONNX model
ONNX model can be generated running the python [code](https://github.com/interscript/arabic-diacritization/blob/master/python/diacritization_model_to_onnx.py) in this library.
It requires to go through some of the steps described in the link above.

### Gems

		 gem install onnxruntime
		 gem install optparse
		 gem install yaml
		 gem install tqdm
