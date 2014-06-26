# -*- coding: utf-8 -*-
from __future__ import print_function
from nltk import download

TOKENIZER_MODEL = "punkt"
POS_TAGGER 		= "maxent_treebank_pos_tagger"

def downloadDependencies():
	download(TOKENIZER_MODEL)
	download(POS_TAGGER)

if __name__ == '__main__':
	downloadDependencies()