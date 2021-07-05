# Arabic Diacritization Library
Arabic diacritization is useful for several practical business cases like text to speech or romanization of arabic texts or scripts.

## Purpose

This repository contains everything to train a diacritization model in python and run it in python and Ruby.

This library was built for the [Interscript project](https://www.interscript.com) ([at GitHub](https://github.com/secryst/secryst)).

Diacritization strategy is following several steps with at heart a deep learning model:
	1. text preprocessing (cutting long sentences into pieces)
	2. neural networks model prediction
	3. rules based solution and search for edge cases (single words like person or geographic names)
	4. text postprocessing
 
This repository contains: 
- [Arabic_Diacritization](https://github.com/interscript/arabic-diacritization/tree/master/Arabic_Diacritization "Arabic_Diacritization")  
	- A **neural network solution** for automatised diacritization based on the work of [almodhfer](https://github.com/almodhfer/Arabic_Diacritization), from which we overtook the baseline and more advanced and efficient CBHG models only. 
This very recent solution allows for efficient predictions on CPU's with a reasonable sized model.
	* **Strings Pre-/Post-processing**, also from [almodhfer](https://github.com/almodhfer/Arabic_Diacritization)
	* **Strings search** within a corpus for single words. Even though NNets capture some understanding of arabic lexicography, single words without contexts can be easily interpreted wrongly. 
* [tests and benchmarking utilities](https://github.com/interscript/arabic-diacritization/tree/master/tests-benchmarks), allowing to compare with other implementations. 
	* tests are are taken from [diacritization benchmarking](https://github.com/AliOsm/arabic-text-diacritization)
	* we have added own, realistic datasets for the problem of diacritization

