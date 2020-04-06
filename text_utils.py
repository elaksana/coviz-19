import re
import nltk

def download_tokenizers(tokenizers):
    ''' download tokenizers if they do not yet exist. 

    input: 
        tokenizers [list(str)]: list of nltk tokenizer names
    '''
    for tokenizer in tokenizers:
        nltk.download(tokenizer)


def sent2wordlist(sentence):
    ''' create tokenized sentences (word-separated list)
    input: 
        sentence [str]: single sentence
    '''
    # remove non-alpha characters
    clean = re.sub('[^a-zA-Z]', " ", sentence)

    # put cleaned sentence into a list of words
    wordlist = clean.split()

    return wordlist

def get_section_text(section_data):
    ''' get text from cord19 article
    '''
    text = u''
    for paragraph_data in section_data:
        text += paragraph_data['text']
    return text