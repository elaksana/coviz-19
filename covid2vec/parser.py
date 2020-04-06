import nltk

class Parser:

    def __init__(self, tokenize_sent):
        ''' Class that tokenizes documents.
        args:
            tokenize_sent [function]:
                function that scrubs text (removes numbers, etc) and returns a list of words (tokens) from that sentence
                input: 
                    sent [str]
                out: 
                    tokenized_sent [list[str]]
        '''

        # set class attributes
        self.tokenize_sent = tokenize_sent

        # download nltk dependencies
        # to split up sentences
        nltk.download('punkt')
        self.punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def tokenize_passage(self, passage):
        ''' Tokenizes passage using class' clean_func function and returns a list of tokenized sentences

        inputs:
            passage [str]:
                string passage
        out:
            tokenized_passage [list(list(str))]:
                list of tokenized sentences
        '''

        tokenized_passage = []

        # tokenize each sentence in a passage, and put them all in a list
        for sentence in self.punkt_tokenizer(passage):
            tokenized_sentence = self.tokenize_sent(sentence)
            tokenized_passage.append(tokenized_sentence)

        return tokenized_passage
