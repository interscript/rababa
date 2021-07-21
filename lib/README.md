# Arabic Diacritization in Ruby with Kithara

## Try out Kithara

* Install the Gems listed below
* Download a ruby model on [releases](https://github.com/secryst/arabic-diacritization-deep-learning-models)

### Run examples
One can diacritize either single strings:

```sh
$ ruby kithara.rb -t 'قطر' -m '../models-data/diacritization_model_max_len_200.onnx'
```

Or files as data/examples.txt or your own arabic file (the max string length is specified in the model and has to match /config/models.yaml's max_len parameter):

```sh
ruby kithara.rb -f 'data/example.txt' -m '../models-data/diacritization_model_max_len_200.onnx'
```

One would have to preprocess generic arabic texts for running Kithara in general. This can be done on sentences beginnings running for instance [Hamza5](https://github.com/Hamza5/Pipeline-diacritizer):
```
python __main__.py preprocess source destination
```


### ONNX Models

They can either be built in the /python repository or downloaded from the
[releases](https://github.com/secryst/arabic-diacritization-deep-learning-models).

Or ONNX model can be generated running the python
[code](https://github.com/interscript/arabic-diacritization/blob/master/python/diacritization_model_to_onnx.py)
in this library.

It requires to go through some of the steps described in the link above.

### Parameters

* text to diacritize: "**-t**TEXT", "--text=TEXT",
* path to file to diacritize: "**-f**FILE", "--text_filename=FILE",
* path to ONNX model **Mandatory**: "-mMODEL", "--model_file=MODEL",
* path to config file **Default:config/model.yml**: "-cCONFIG", "--config=CONFIG"

### Config

#### Players:

* max_len: 200 -- 600
	* Parameter that has to match the ONNX model built using the
	  [code]{https://github.com/interscript/arabic-diacritization/blob/master/python/diacritization_model_to_onnx.py}
	  and following the python/Readme.md.
	* Longer sentences will need to be preprocessed, which can be done for
	  instance using [Hamza5](https://github.com/Hamza5)
	  [code](https://github.com/Hamza5/Pipeline-diacritizer/blob/master/pipeline_diacritizer/pipeline_diacritizer.py).
	* the smaller the faster the nnets code.
* text_encoder corresponding to the [rules](https://github.com/interscript/arabic-diacritization/blob/master/python/util/text_encoders.py):
     * BasicArabicEncoder
     * ArabicEncoderWithStartSymbol
* text_cleaner corresponding to [logics](https://github.com/interscript/arabic-diacritization/blob/master/python/util/text_cleaners.py):
     * basic_cleaners: default
     * valid_arabic_cleaners: the code would need to be massaged at the encoder's instantiation level to activate existing cleaner function.

### Gems

```sh
gem install onnxruntime
gem install optparse
gem install yaml
gem install tqdm
```
