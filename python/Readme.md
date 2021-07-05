# Diacritization Model




## Core: Python Implementation of several deep learning models for recovering Arabic language diacritics

We are refering here to the [code](https://github.com/almodhfer/Arabic_Diacritization) and 
[Effective Deep Learning Models for Automatic Diacritization of Arabic Text](https://ieeexplore.ieee.org/document/9274427) that we have selected for this project from a list of alternatives listed in the docs readme.

Out of the four models that [almodhfer](https://github.com/almodhfer) has implemented, we selected the simplest and most performant ones:

- The baseline model (baseline): consists of 3 bidirectional LSTM layers with optional batch norm layers.
- The CBHG model (cbhg): uses only the encoder of the Tacotron based model with optional post LSTM, and batch norm layers.


### Datasets

- We have chosen the Tashkeela corpus ~2800000 sentences:
    * [huggingface](https://huggingface.co/datasets/tashkeela)
    * [sourceforge](https://sourceforge.net/projects/tashkeela-processed/)
```bash
    mkdir data
    mkdir data/CA_MSA
    unzip data.zip
    mv ~* data/CA_MSA/.
    > ls data/CA_MSA/*
        --> data/CA_MSA/eval.csv  data/CA_MSA/train.csv  data/CA_MSA/test.csv
```
    

### Data Preprocessing

- The data can either be processed before training or when loading each batch.
- If you decide to process the corpus before training, then the corpus must have test.csv, train.csv, and valid.csv. Each file must contain three columns: text (the original text), text without diacritics, and diacritics. You have to define the column separator and diacritics separator in the config file.
- If the data is not preprocessed, you can specify that in the config.
  In that case,  each batch will be processed and the text and diacritics 
  will be extracted from the original text.
- You also have to specify the text encoder and the cleaner functions.
  This work includes two text encoders: BasicArabicEncoder, ArabicEncoderWithStartSymbol.
  Moreover, we have one cleaning function: valid_arabic_cleaners, which clean all characters except valid Arabic characters,
  which include Arabic letters, punctuations, and diacritics.

### Training

All models config are placed in the config directory.

```bash
python train.py --model model_name --config config/config_name.yml
```

The model will report the WER and DER while training using the
diacritization_evaluation package. The frequency of calculating WER and
DER can be specified in the config file.

### Testing

The testing is done in the same way as the training:

```bash
python test.py --model model_name --config config/config_name.yml
```

The model will load the last saved model unless you specified it in the config:
test_data_path. If the test file name is different than test.csv, you
can add it to the config: test_file_name.


### Conversion of CBHG to ONNX

The last model stored during traing is automatically chosen and the onnx model is saved into a hardcoded location:
"../models-data/diacritization_model.onnx"

```bash
python diacritization_model_to_onnx.py
```

## Preprocessing in Python


## Postprocessing in Python


## Search for single strings in Python & Business rules


## Python to Ruby

* We are using the following libraries:
* We rewrote the following components:


