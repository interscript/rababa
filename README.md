# رُبابَة RABABA an Arabic Diacritization Library

Arabic diacritization is useful for several practical business cases like text
to speech or Romanization of Arabic texts or scripts.

## Purpose

This repository contains everything to train a diacritization model in Python
and run it in Python and Ruby.

## Try out Rababa

Rababa can be run both in python and ruby. Go the directory corresponding to the language you prefer to use.  Indications are in the README's, under the "Try out Rababa" section:
* [Python](https://github.com/interscript/rababa/tree/master/python)
* [Ruby](https://github.com/interscript/rababa/tree/master/lib)

## Library

This library was built for the
[Interscript project](https://www.interscript.com)
([at GitHub](https://github.com/interscript/)).

Diacritization strategy is following several steps with at heart a deep learning
model:

1. text preprocessing
2. neural networks model prediction
3. text postprocessing

This repository contains:

- [lib](https://github.com/interscript/rababa/tree/master/lib) is
  the Ruby library using NNet model in ONNX format.

- [docs](https://github.com/interscript/rababa/tree/master/docs)
  contains an application focused summary of latest (2021-06) relevant papers
  and solutions.

- [python](https://github.com/interscript/rababa/tree/master/python)
    - A **neural network solution** for automatised diacritization based on the
      work of [almodhfer](https://github.com/almodhfer/Arabic_Diacritization),
      from which we overtook the baseline and more advanced and efficient CBHG
      models only. This very recent solution allows for efficient predictions on
      CPU's with a reasonable sized model.

    * **PyTorch to ONNX** conversion of PyTorch to ONNX format

    * **Strings Pre-/Post-processing**, also from
      [almodhfer](https://github.com/almodhfer/Arabic_Diacritization)

- [tests and benchmarking utilities](https://github.com/interscript/rababa/tree/master/tests-benchmarks),
  allowing to compare with other implementations.

	* tests are are taken from
	  [diacritization benchmarking](https://github.com/AliOsm/arabic-text-diacritization)

	* we have added own, realistic datasets for the problem of diacritization

- **models-data** directory to store models and embeddings in various formats

## About the Name
A Rababa is an antic string instrument.
In a similar fashion that a Rababa produces melody from a simple strings and pieces of wood,
our library and diacritization gives a whole palette of colour and meanings to arabic scripts.

## Under development
We are working on the following improvements:
* Preprocessing for breaking down large sentences
* PoS tagging and search to improve the diacritization
