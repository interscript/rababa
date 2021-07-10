# Diacritization Model




## Core: Python Deep Learning models for recovering Arabic language diacritics

We are refering here to the [code](https://github.com/almodhfer/Arabic_Diacritization) and 
[Effective Deep Learning Models for Automatic Diacritization of Arabic Text](https://ieeexplore.ieee.org/document/9274427) that we have selected for this project from a list of alternatives listed in the docs readme.

Out of the four models that [almodhfer](https://github.com/almodhfer) has implemented, we selected the simplest and most performant ones:

- The baseline model (baseline): consists of 3 bidirectional LSTM layers with optional batch norm layers.
- The CBHG model (cbhg): uses only the encoder of the Tacotron based model with optional post LSTM, and batch norm layers.

### Requirements
```bash
pip install -r requirement.txt
```

### Datasets

- We have chosen the Tashkeela corpus ~2800000 sentences:
    * [huggingface](https://huggingface.co/datasets/tashkeela)
    * [sourceforge](https://sourceforge.net/projects/tashkeela-processed/)
```bash
    mkdir data
    mkdir data/CA_MSA
    unzip data.zip 
```
for training, data need to be in format:
```bash
    > ls data/CA_MSA/*
        --> data/CA_MSA/eval.csv  data/CA_MSA/train.csv  data/CA_MSA/test.csv
```
so for instance:
```bash
for d in `ls tashkeela_val/*`; do; cat $d >> data/CA_MSA/eval.csv; done
for d in `ls tashkeela_train/*`; do; cat $d >> data/CA_MSA/train.csv; done
for d in `ls tashkeela_test/*`; do; cat $d >> data/CA_MSA/test.csv; done

```

### Load Model

Models are available under 
[releases](https://github.com/secryst/arabic-diacritization-deep-learning-models).
Models are to be copied under /log_dir/ as specified in the link just above.

### Config Files
One can adjust the model configurations in the /config repository.
The model configurations are about the layers but also the dataset to be used and various other options.
The configuration files are called explicitely in the below applications.

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

### "Diacritize" Text or Files

Single sentences or files can be processed. The code outputs is the diacretized text or lines.
```bash
python diacritize.py --model_kind "cbhg" --config config/cbhg.yml --text 'قطر'
python diacritize.py --model_kind "cbhg" --config config/cbhg.yml --text_file relative_path_to_text_file
```


### Convert CBHG, python model to ONNX

The last model stored during traing is automatically chosen and the onnx model is saved into a hardcoded location:
"../models-data/diacritization_model.onnx"

```bash
python diacritization_model_to_onnx.py
```

# TODO List


## Bug with diacritics combination


## Preprocessing in Python


## Postprocessing in Python


## Search for single strings in Python & Business rules

## Full code in ruby + onnx

## Python to Ruby

* We are using the following libraries:
* We rewrote the following components:

...
