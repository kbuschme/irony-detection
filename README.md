# System for Irony Detection in Product Reviews


## Dependencies

A working [Python](https://www.python.org/) installation (tested with version 2.7.5)
and the following Python libraries are needed:

* [NumPy](http://www.numpy.org)
* [SciPy](http://www.scipy.org)
* [scikit-learn](http://www.scikit-learn.org/)
* [Natural Language Toolkit](http://www.nltk.org/) (NLTK) and the following models:
    
    * Treebank Part of Speech Tagger (HMM) - hmm\_treebank\_pos\_tagger
    * Punkt Tokenizer Models - punkt


        To download models follow these steps:

        1. Start Python in a terminal:

            > python

        2. Start the graphical user interface and download the listed models:

        	> import nltk

        	> nltk.download()


### Required files

* [Sarcasm Corpus](http://storm.cis.fordham.edu/filatova/SarcasmCorpus.html) by Elena Filatova[1].
* [Opinion Lexicon](http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) by Hu and Liu[2].



## Installation

1. Make sure your system meets the dependencies listed above.
2. Download the System from github.

3. Add the Sarcasm Corpus
	1. Unpack the archive `SarcasmCorpus.rar` and 
	2. place the unpacked folder `SarcasmCorpus` in the directory 
    > ../Corpora/

4. Add the opinion lexicon
	1. Unpack the archive `opinion-lexicon-English.rar` and 
	2. place the files `negative-words.txt` and `positive-words.txt` from the upacked folder `opinion-lexicon-English` in the directory
    > ../src/ 


## Getting Started

    python main.py --help


## References

[1] 

[2] 

[3]


---

Copyright (C) Konstantin Buschmeier.