# System for Irony Detection in Product Reviews

This system was used for the paper 
[An impact analysis of features in a classification approach to irony detection in product reviews](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf)[1].


## Installation

### 1. Download the system

Download the system with the following command

```
> git clone https://github.com/kbuschme/irony-detection.git
```


### 2. Download the corpus

Download the Sarcasm Corpus by Elena Filatova[2] from its [Github repository](https://github.com/ef2020/SarcasmAmazonReviewsCorpus) and place it inside the `corpora` directory

```
> git clone https://github.com/ef2020/SarcasmAmazonReviewsCorpus.git corpora/SarcasmCorpus
```

Alternatively download the [Zip archive](https://github.com/ef2020/SarcasmAmazonReviewsCorpus/archive/master.zip) and place the extracted files in the `corpora` directory.

Unpack the archive `Ironic.rar` into a directory `corpora/SarcasmCorpus/Ironic` and the archive `Regular.rar` into a directory `corpora/SarcasmCorpus/Regular`

```
> unrar e corpora/SarcasmCorpus/Ironic.rar corpora/SarcasmCorpus/Ironic/
> unrar e corpora/SarcasmCorpus/Regular.rar corpora/SarcasmCorpus/Regular/
```


### 3. Download additional resources

Download the file `opinion-lexicon-English.rar` which contains the [Opinion Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) by Hu and Liu[3] and place it inside the `resources` directory

```
> curl -o resources/opinion-lexicon-English.rar https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
```

Unpack the archive `opinion-lexicon-English.rar` and place the files `negative-words.txt` and `positive-words.txt` inside the `resources` directory

```
> unrar e resources/opinion-lexicon-English.rar resources/
```


### 4. Install python libraries and language models

A working [Python](https://www.python.org/) 2 installation (tested with version 2.7.14)
and the following Python libraries are needed. These can be installed using [pip](https://pypi.python.org/pypi/pip):

* [NumPy](http://www.numpy.org) (version 1.14.0)

    ```
    > sudo pip install numpy==1.14.0
    ```

* [SciPy](http://www.scipy.org) (version 1.0.0)

    ```
    > sudo pip install scipy==1.0.0
    ```

* [scikit-learn](http://www.scikit-learn.org/) (version 0.19.1)

    ```
    > sudo pip install scikit-learn==0.19.1
    ```

* [pydot](https://github.com/erocarrera/pydot) (version 1.2.4) with [pyparsing](http://pyparsing.wikispaces.com) (version 2.2.0)

    ```
    > sudo pip install pyparsing==2.2.0
    > sudo pip install pydot==1.2.4
    ```

* [Natural Language Toolkit](http://www.nltk.org/) (NLTK) (version 3.2.5) with [PyYAML](https://pyyaml.org) (version 3.12)

    ```
    > sudo pip install PyYAML==3.12
    > sudo pip install nltk==3.2.5
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
usage: Irony Detector [-h] {corpus,feature,interactive,ml,sets} ...
Irony Detector: error: too few arguments
```

The following commands are available and described in the Manual section below:

* `corpus`,
* `features`,
* `interactive`,
* `ml` and
* `sets`

As a first step the `sets` command should be run. This will create three files inside the `corpora/SarcasmCorpus` directory. The file `shuffled_set.txt` is a randomized version of the corpus used for cross-validation. The files `training_set.txt` and `test_set.txt` are a training and test set and contain 90% and 10% of the reviews, respectively.

```
> python main.py sets
```

Now the *machine learning mode* of the system can be used to classify reviews. The following example applies 10-fold cross-validation:

```
> python main.py ml cross-validation
```


## Manual

### Help

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
usage: Irony Detector [-h] {corpus,feature,interactive,ml,sets} ...

Detects irony in amazon reviews.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  The following commands can be invoked.

  {corpus,feature,interactive,ml,sets}
                        Valid commands.
    corpus              Show details about the entire corpus.
    feature             Shows how often each feature is found for ironic and
                        regular reviews in the training_and_validation_set.
    interactive         The interactive mode classifies a given sentence using
                        a saved model.
    ml                  Use the machine learning approach to classify reviews.
    sets                Divide the corpus into training, validation and test
                        set.
```


### Corpus mode

The *corpus mode* shows general information about a corpus.

Show all reviews inside the corpus:

```
> python main.py corpus reviews
```

Show some statistics about the corpus:

```
> python main.py corpus stats
```


### Feature mode

The *feature mode* displays statistics about the specific features or exports all features as Attribute-Relation File Format[](http://www.cs.waikato.ac.nz/ml/weka/arff.html) (ARFF).

Show how often the specific features occur in all reviews:

```
> python main.py feature show
```

Export the extracted feature to an ARFF file:

```
> python main.py feature export
```


### Machine learning mode

The *machine learning mode* uses the following classifiers to classify the reviews:

* Naive Bayes,
* Decision Tree,
* Random Forest,
* Logistic Regression and
* Support Vector Machine

Use 10-fold cross-validation:

```
> python main.py ml cross-validation
```

Train the classifiers on a training set and classify a test set:

```
> python main.py ml test
```


### Sets mode

On one hand the *sets mode* generates a shuffled set for cross-validation
and on the other hand divides all reviews into a training and test set by a 90 to 10 ratio.

```
> python main.py sets
```

This command creates the following three files inside the directory `corpora/SarcasmCorpus`: 

* `corpora/SarcasmCorpus/shuffled_set.txt`,
* `corpora/SarcasmCorpus/training_set.txt` and
* `corpora/SarcasmCorpus/test_set.txt`.


## References

[1] Konstantin Buschmeier, Philipp Cimiano, and Roman Klinger. [An impact analysis of features in a classification approach to irony detection in product reviews](http://acl2014.org/acl2014/W14-26/pdf/W14-2608.pdf). In *Proceedings of the 5th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*, pages 42–49, Baltimore, Maryland, June 2014. Association for Computational Linguistics.

[2] Elena Filatova. 2012. [Irony and sarcasm: Corpus Generation and Analysis Using Crowdsourcing](http://storm.cis.fordham.edu/~filatova/PDFfiles/FilatovaLREC2012.pdf). In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Mehmet Uğur Doğan, Bente Maegaard, Joseph Mariani, Jan Odijk, and Stelios Piperidis, editors, *Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC-2012)*, pages 392–398, Istanbul, Turkey, May. European Language Resources Association (ELRA).

[3] Minqing Hu and Bing Liu. 2004. [Mining and summarizing customer reviews](http://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf). In *Proceedings of the tenth ACM SIGKDD international conference of Knowledge discovery and data mining, KDD '04*, pages 168–177, New York, NY, USA. ACM.


---

Copyright (C) Konstantin Buschmeier.
