# Literature and Codes

Last updated: 2021-06.

We review only the more advanced technologies.

Older solutions used rules based approaches.

Deep Learning was applied relatively to the problem of diacritization, gradually
getting better results than rules based approaches.

**Mishkal, Arabic text vocalization software**
Zerrouki, T.
 rules based library, 2014
 * [code](https://github.com/linuxscout/mishkal)

**Automatic minimal diacritization of Arabic texts**
Rehab Alnefaiea, Aqil M.Azmib
11.2017
* MADAMIRA software
* [paper](https://www.sciencedirect.com/science/article/pii/S1877050917321634)

**An Approach for Arabic Diacritization**
 Ismail Hadjir, Mohamed Abbache, Fatma Zohra Belkredim
06.2019
* keywords: Hidden Markov Models, Viterbi algorithm
* [article](https://link.springer.com/chapter/10.1007/978-3-030-23281-8_29)

**Diacritization of Moroccan and Tunisian Arabic Dialects: A CRF Approach**
Kareem Darwish∗, Ahmed Abdelali∗, Hamdy Mubarak∗, Younes Samih†, Mohammed Attia⋆
2018
* keywords: Conditional Random Fields, arabic dialects...
* [paper](http://lrec-conf.org/workshops/lrec2018/W30/pdf/20_W30.pdf)

**Arabic Text Diacritization Using Deep Neural Networks**
Ali Fadel, Ibraheem Tuffaha, Bara' Al-Jawarneh, Mahmoud Al-Ayyoub
**Shakkala** library, tensorflow,  04.2019
* keywords: Embedding, LSTM
*  [paper](https://arxiv.org/abs/1905.01965)
*  [code](https://github.com/Barqawiz/Shakkala), tensorflow
* [benchmarks&scripts](https://github.com/AliOsm/arabic-text-diacritization)

**Highly Effective Arabic Diacritization using Sequence to Sequence Modeling**
* Hamdy Mubarak, Ahmed Abdelali, Hassan Sajjad, Younes Samih, Kareem Darwish
06.2019
* keywords: seq2seq(LSTM), NMT, interesting representation units, context window, voting
* [paper](https://www.aclweb.org/anthology/N19-1248.pdf)

**Multi-components System for Automatic Arabic Diacritization**
Hamza Abbad, Shengwu Xiong
04.2020
* keywords: LSTM's, parallel layers for Shadda and Harakat (⇒ pipeline)
* [paper](https://paperswithcode.com/paper/multi-components-system-for-automatic-arabic)
* [code](https://github.com/Hamza5/Pipeline-diacritizer), tensorflow

**Deep Diacritization: Efficient Hierarchical Recurrence for Improved Arabic Diacritization**
Badr AlKhamissi, Muhammad N. ElNokrashy, and Mohamed Gabr
12.2020
* keywords: Cross-level attention, Encoder-Decoder (LSTM), Teacher forcing,
* [paper](https://www.aclweb.org/anthology/2020.wanlp-1.4.pdf)
* [slides](https://drive.google.com/file/d/1GzXRIddVeJRCge74QaRC67M1I-pAoGV3/view)
* [code](https://github.com/BKHMSI/deep-diacritization), pytorch

**Effective Deep Learning Models for Automatic Diacritization of Arabic Text**
Mokthar Ali Hasan Madhfar; Ali Mustafa Qamar
12.2020
* keywords: embedding, encoder-decoder (LSTM), Highway Nets, Attention, CBHG Module
* [paper](https://paperswithcode.com/paper/effective-deep-learning-models-for-automatic)
* [code](https://github.com/almodhfer/Arabic_Diacritization), pytorch

**A Deep Belief Network Classification Approach for Automatic Diacritization of Arabic Text**
Mohammad Aref Alshraideh, Mohammad Alshraideh and Omar Alkadi
4.2021
* keywords: DBN built with Boltzmann restricted machines (restricted RBM's) superior to LSTMs, unicode encoding, Borderline-SMOTE
* [paper](https://www.researchgate.net/publication/352226815_A_Deep_Belief_Network_Classification_Approach_for_Automatic_Diacritization_of_Arabic_Text)


# Research ideas
Here we just mention some 2021-ish ideas mentioned in the recent papers above:
* Transformer-based Encoders
* Byte-pair-encodings
* Improve Injected Hints Method (train with semi diacritised data)
* More Interpretable Attention Weights
* Deep belief networks
* More data and data processing
