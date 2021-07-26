# Arabic Diacritization in Ruby with Rababa

## Try out Rababa

* Install the Gems listed below
* Download a ruby model on [releases](https://github.com/secryst/rababa-models)

### Run examples

Prerequisite:

* Please download the `diacritization_model_max_len_200.onnx` model file
from https://github.com/secryst/rababa-models/releases/tag/0.1

One can diacritize either single strings:

```sh
rababa -t 'قطر' -m diacritization_model_max_len_200.onnx
# or when inside the gem directory during development
bundle exec exe/rababa -t 'قطر' -m diacritization_model_max_len_200.onnx
```

Or files as `data/examples.txt` or your own Arabic file (the max string length
is specified in the model and has to match the `max_len` parameter in
`config/models.yaml`):

```sh
rababa -f data/example.txt -m diacritization_model_max_len_200.onnx
# or when inside the gem directory during development
bundle exec exe/rababa -f data/example.txt -m diacritization_model_max_len_200.onnx
```

One would have to preprocess generic arabic texts for running Rababa in general.
This can be done on sentences beginnings running for instance
[Hamza5](https://github.com/Hamza5/Pipeline-diacritizer):

```
python __main__.py preprocess source destination
```

### ONNX Models

They can either be built in the `/python` repository or downloaded from the
[releases](https://github.com/secryst/rababa-models).

Or ONNX model can be generated running the python
[code](https://github.com/interscript/rababa/blob/master/python/diacritization_model_to_onnx.py)
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
	  [code]{https://github.com/interscript/rababa/blob/master/python/diacritization_model_to_onnx.py}
	  and following the python/Readme.md.
	* Longer sentences will need to be preprocessed, which can be done for
	  instance using [Hamza5](https://github.com/Hamza5)
	  [code](https://github.com/Hamza5/Pipeline-diacritizer/blob/master/pipeline_diacritizer/pipeline_diacritizer.py).
	* the smaller the faster the nnets code.
* text_encoder corresponding to the [rules](https://github.com/interscript/rababa/blob/master/python/util/text_encoders.py):
     * BasicArabicEncoder
     * ArabicEncoderWithStartSymbol
* text_cleaner corresponding to [logics](https://github.com/interscript/rababa/blob/master/python/util/text_cleaners.py):
     * basic_cleaners: remove redundancy in whitespaces and strip string
     * valid_arabic_cleaners: basic+filter of only arabic words

### Gems

```sh
gem install rababa
```
