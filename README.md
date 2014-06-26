# System for Irony Detection in Product Reviews

**Please note**: The system still needs some clean-up, but will be available soon.



This system was used for the paper 
[An impact analysis of features in a classification approach to irony detection in product reviews](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf)[1].


## Installation


### 1. Download the system
Download the system with the following command
```
> git clone https://github.com/kbuschme/irony-detection.git
```


### 2. Download the corpus

Download the file `SarcasmCorpus.rar` which contains the [Sarcasm Corpus](http://storm.cis.fordham.edu/filatova/SarcasmCorpus.html) by Elena Filatova[2] and place it inside the `corpora` directory
```
> curl -o corpora/SarcasmCorpus.rar http://storm.cis.fordham.edu/~filatova/SarcasmCorpus.rar
```

Unpack the content of the archive `SarcasmCorpus.rar` into a directory `corpora/SarcasmCorpus`
```
> unrar e corpora/SarcasmCorpus.rar corpora/SarcasmCorpus/
```

Unpack the archive `Ironic.rar` into a directory `corpora/SarcasmCorpus/Ironic` and the archive `Regular.rar` into a directory `corpora/SarcasmCorpus/Regular`
```
> unrar e corpora/SarcasmCorpus/Ironic.rar corpora/SarcasmCorpus/Ironic/
> unrar e corpora/SarcasmCorpus/Regular.rar corpora/SarcasmCorpus/Regular/
```


### 3. Download additional resources

Download the file `opinion-lexicon-English.rar` which contains the [Opinion Lexicon](http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) by Hu and Liu[3] and place it inside the ``resources`` directory
```
> curl -o resources/opinion-lexicon-English.rar http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
```

Unpack the files negative-words.txt `opinion-lexicon-English.rar`
```
> unrar e resources/opinion-lexicon-English.rar resources/
```


### 4. Install python libraries and language models

A working [Python](https://www.python.org/) 2 installation (tested with version 2.7.5)
and the following Python libraries are needed. These can be installed using [pip](https://pypi.python.org/pypi/pip):

* [NumPy](http://www.numpy.org)
```
> sudo pip install numpy
```

* [SciPy](http://www.scipy.org)
```
> sudo pip install scipy
```

* [scikit-learn](http://www.scikit-learn.org/)
```
> sudo pip install scikit-learn
```

* [Natural Language Toolkit](http://www.nltk.org/) (NLTK)
```
> sudo pip install PyYAML
> sudo pip install nltk
```

Additionally NLTK requires the following models:

* Max Entropy Pos Tagger (maxent\_treebank\_pos\_tagger) and
* Punkt Tokenizer Models (punkt)

which can be downloaded with the  ``setup.py`` script
```
> python setup.py
```

or manually with the following steps:
```
> python
>>> import nltk
>>> nltk.download("punkt")
>>> nltk.download("maxent_treebank_pos_tagger")
>>> exit()
```





## Getting Started

To start the system change the directory to `src` and run the file `main.py` which provides a command-line interface:
```
> cd src
> python main.py
```

the output should look as follows
```
> python main.py
usage: Irony Detector [-h] {corpus,feature,interactive,ml,rules,sets,test} ...
Irony Detector: error: too few arguments
>
```

The following commands are available and described in the Manual section below:

* corpus
* features
* interactive
* ml
* rules
* sets
* test



## Manual

### Help:
Show a short help message about the available commands:
```
> python main.py -h
```

Show a detailed help message about the available commands:
```
> python main.py --help
```

This should look like the following message:
```
usage: Irony Detector [-h] {corpus,feature,interactive,ml,rules,sets,test} ...

Detects irony in amazon reviews.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  The following commands can be invoked.

  {corpus,feature,interactive,ml,rules,sets,test}
                        Valid commands.
    corpus              Show details about the entire corpus.
    feature             Shows how often each feature is found for ironic and
                        regular reviews in the training_and_validation_set.
    interactive         The interactive mode classifies a given sentence using
                        a saved model.
    ml                  Use the machine learning approach to classify reviews.
    rules               Use the rule based approach to classify reviews.
    sets                Divide the corpus into training, validation and test
                        set.
    test                Test basic functionality of the application.
```



## References

[1] Konstantin Buschmeier, Philipp Cimiano, and Roman Klinger. [An impact analysis of features in a classification approach to irony detection in product reviews](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf). In *Proceedings of the 5th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*, pages 42–49, Baltimore, Maryland, June 2014. Association for Computational Linguistics.

[2] Elena Filatova. 2012. [Irony and sarcasm: Corpus Generation and Analysis Using Crowdsourcing](http://storm.cis.fordham.edu/~filatova/PDFfiles/FilatovaLREC2012.pdf). In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Mehmet Uğur Doğan, Bente Maegaard, Joseph Mariani, Jan Odijk, and Stelios Piperidis, editors, *Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC-2012)*, pages 392–398, Istanbul, Turkey, May. European Language Resources Association (ELRA).

[3] Minqing Hu and Bing Liu. 2004. [Mining and summarizing customer reviews](http://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf). In *Proceedings of the tenth ACM SIGKDD international conference of Knowledge discovery and data mining, KDD '04*, pages 168–177, New York, NY, USA. ACM.


---

Copyright (C) Konstantin Buschmeier.