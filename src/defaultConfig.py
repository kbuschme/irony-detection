# -*- coding: utf-8 -*-
# ==== Files and Paths ====
# ---- Polarity resources: ----
NEGATIVE_WORDS_FILENAME = "../resources/negative-words.txt"
POSITIVE_WORDS_FILENAME = "../resources/positive-words.txt"

# ---- Corpus resources: ----
CORPUS_PATH = "../corpora/SarcasmCorpus/"
IRONIC_REVIEWS_PATH = "Ironic/"
REGULAR_REVIEWS_PATH = "Regular/"

REVIEW_IDS_FILENAME = "file_pairing.txt"
IRONIC_UTTERANCES_FILENAME = "sarcasm_lines.txt"

# ---- Training, valdiation and test set: ----
WORKING_COPY_PATH = "../corpora/WorkingCopy/"
SET_FILENAMES = ["training_set.txt",
                "test_set.txt", 
                "validation_set.txt",
                "shuffled_set.txt"]

ARFF_FILENAME = "features.arff"

TRAINING_SET_SIZE = 80
VALIDATION_SET_SIZE = 10
TEST_SET_SIZE = 10
RANDOM_SEED = 44



# ==== Features and their configurations ====
REGEX_FEATURE_CONFIG = {
    # ---- Emoticons ----
    u"Emoticon Happy": (u":-)", 
                        r"""[:=][o-]?[)}>\]]|               # :-) :o) :)
                        [({<\[][o-]?[:=]|                   # (-: (o: (:
                        \^(_*|[-oO]?)\^                     # ^^ ^-^
                        """
    ), 
    u"Emoticon Laughing": (u":-D", r"""([:=][-]?|x)[D]"""),   # :-D xD
    u"Emoticon Winking": (u";-)", 
                        r"""[;\*][-o]?[)}>\]]|              # ;-) ;o) ;)
                        [({<\[][-o]?[;\*]                   # (-; (
                        """
    ), 
    u"Emotion Tongue": (u":-P", 
                        r"""[:=][-]?[pqP](?!\w)|            # :-P :P
                        (?<!\w)[pqP][-]?[:=]                # q-: P-:
                        """
    ),  
    "Emoticon Surprise": (u":-O", 
                            r"""(?<!\w|\.)                  # Boundary
                            ([:=]-?[oO0]|                   # :-O
                            [oO0]-?[:=]|                    # O-:
                            [oO](_*|\.)[oO])                # Oo O____o O.o
                            (?!\w)
                            """
    ), 
    u"Emoticon Dissatisfied": (u":-/", 
                                r"""(?<!\w)                 # Boundary
                                [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
                                [\/\\|][-o]?[:=]|           # \-: \:
                                -_+-                        # -_- -___-
                                """
    ), 
    u"Emoticon Sad": (u":-(", 
                        r"""[:=][o-]?[({<\[]|               # :-( :(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?[:=]                    # )-: ): )o: 
                        """
    ), 
    u"Emoticon Crying": (u";-(", 
                        r"""(([:=]')|(;'?))[o-]?[({<\[]|    # ;-( :'(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?(('[:=])|('?;))         # )-; )-';
                        """
    ), 
    
    # ---- Punctuation----
    # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
    u"Question Mark": (u"??", r"""\?{2,}"""),                 # ??
    u"Exclamation Mark": (u"!!", r"""\!{2,}"""),              # !!
    u"Question and Exclamation Mark": (u"?!", r"""[\!\?]*((\?\!)+|              # ?!
                    (\!\?)+)[\!\?]*                         # !?
                    """
    ),                                          # Unicode interrobang: U+203D
    u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
                \.(\ \.){2,}                                # . . .
                """
    ),                                          # Unicode Ellipsis: U+2026
    # ---- Markup----
    u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
    u"Pseudo-Tag": (u"Tag", 
                    r"""([<\[][\/\\]
                    (irony|ironic|sarcasm|sarcastic)        # </irony>
                    [>\]])|                                 #
                    ((?<!(\w|[<\[]))[\/\\]                  #
                    (irony|ironic|sarcasm|sarcastic)        # /irony
                    (?![>\]]))
                    """
    ),

    # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
    u"Onomatopoeia for Laughter": (u"haha", 
                        r"""(?<!\w)                         # Boundary
                        (mu|ba)?                            # mu- ba-
                        (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
                        (?!\w)                              # Boundary
                        """
    ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
}



    # Pseudo-Tag regEx:
    # r"(\<|\[)?(\/|\\)(sarcasm|sarcastic|irony|ironic)(\>|\])?"
    # r"([<\[][\/\\](irony|ironic|sarcasm|sarcastic)[>\]])|((?<!(\w|[<\[]))[\/\\](irony|ironic|sarcasm|sarcastic)(?![>\]]))"

    # Emoticons:
    # "EmoSurprise": (":-O", 
    #                 r"""[:=]-[oO0]|                # :-O
    #                 [oO0]-[:=]|                   # O-:
    #                 [oO0](_*|.)[oO0]\W?           # Oo oO O_o
    #                 """,              # Old style, all with "oo", that is BAD!

    # Carvalho:
    # Best patterns: Pijt, Ppunct, Pquote, Plaugh
    #
    # Pdim        (4-Gram+ NEdim | NEdim 4-Gram+)
    # Pdem        DEM NE 4-Gram+
    # Pitj        ITJpos (DEM ADJpos)∗ NE (?|!|...)
    # Pverb       NE (tu)∗ ser2s 4-Gram+
    # Pcross      (DEM|ART) (ADJpos|ADJneut) de NE
    # Ppunct      4-Gram+ (!!|!?|?!)
    # Pquote      “(ADJpos|Npos){1, 2}”
    # Plaugh      (LOL|AH|EMO+ )
    # scare quotes / air quotes