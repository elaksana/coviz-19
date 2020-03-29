import os

## IO Stuff ##
# word encoding
import codecs
#json 
import json


# file pathnames
import glob

# regex
import re

# natural language toolkit
import nltk

# word2vec and doc2vec model
import gensim.models.word2vec as w2v
import gensim.models.doc2vec as d2v

# dimensionality reduction algorithms
import sklearn.manifold

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# pandas
import pandas as pd

# custom utils function to make my life easier
import utils
import pdb


MODEL2ID_DICT = {
    'cbow': 0,
    'skipgram': 1
}

def sent2wordlist(sentence):
    # remove non-alpha characters
    clean = re.sub('[^a-zA-Z]', " ", sentence)

    # put cleaned sentence into a list of words
    wordlist = clean.split()

    return wordlist

def get_section_text(section_data):
    text = u''
    for paragraph_data in section_data:
        text += paragraph_data['text']
    return text

## Download nltk dependencies ##
nltk.download('punkt')
nltk.download('stopwords')

# tokenizer used to split sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# output directory setup
if not os.path.exists('../models'):
    os.makedirs('../models')


##### File IO #####
# 1. read json files in corpus
# 2. store all abstract and body text into a single raw string
# 3. tokenize strings
DS_DIR = '/home/data/CORD-19'

# get all json filepaths
json_filepaths = glob.glob(os.path.join(DS_DIR, '*', '*', '*.json'))


doc_df = {
    'body': [],
    'abstract': [],
    'title': [],
    'sha': []
    }

# raw_corpus = u''

# combine papers into one file
for filepath in json_filepaths[0:10]:

    # load json data
    json_data = utils.load_json(filepath)

    title = json_data['metadata']['title']

    # paper id
    sha = json_data['paper_id']
    # extract text from paper's body
    
    abstract = get_section_text(json_data['abstract'])
    body = get_section_text(json_data['body_text'])

    doc_df['title'].append(title)
    doc_df['sha'].append(sha)

    abstract_ps = []
    # create a list (words) of lists (sentences) for abstract and body text
    for sentence in tokenizer.tokenize(abstract):
        #print(sentence)
        abstract_ps.append(sent2wordlist(sentence))
    doc_df['abstract'].append(abstract_ps)
    
    body_ps = []
    for sentence in tokenizer.tokenize(body):
        body_ps.append(sent2wordlist(sentence))
    doc_df['body'].append(body_ps)

doc_df = pd.DataFrame(doc_df)
doc_df['full_text'] = doc_df['abstract'] + doc_df['body']

full_tokenized_corpus = [sent for doc in doc_df['full_text'].values for sent in doc]
print('Dataset contains %i tokens.' % (sum([len(full_tokenized_corpus) for sentence in full_tokenized_corpus])))


# ##### Train word2vec model #####
# word2vec parameters

# number of high-dim features
num_features = 100

# minimum count occurances to show up in model
min_word_count = 5

# number of parallel processes
num_workers = 1

# context window length
context_size = 7

# downsample high-frequency words
downsampling = 1e-3

# rseed for reproducibility
seed = 3

# use a skipgram model (set to 0 for cbow)
model = 'skipgram'

covid2vec = w2v.Word2Vec(
    sg=MODEL2ID_DICT[model],
    seed=seed, 
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
    )

# build vocabular
covid2vec.build_vocab(full_tokenized_corpus)

print('Vocab Length: %i' % len(covid2vec.wv.vocab))

# train
covid2vec.train(full_tokenized_corpus, total_examples=covid2vec.corpus_count, epochs=covid2vec.epochs)

# save my model
covid2vec.save(os.path.join('..','models','covid2vec.w2v'))

# ## DOCUMENT-LEVEL EMBEDDING ##
# 1. store word embeddings into pandas dataframe for simple lookup
# 2. get intradocument word vector average
word_embedding_matrix = covid2vec.wv.vectors
word_embedding_df = pd.DataFrame(word_embedding_matrix)
word_embedding_df['word'] = covid2vec.wv.vocab
word_embedding_df = word_embedding_df.set_index('word')
def vectorize_doc(row):
    ''' Creates vectorized embedding of a document by averaging vectorized words
    '''
    words = [sent for doc in row for sent in doc]
    return word_embedding_df.reindex(words).mean()

doc_df = pd.concat([doc_df, doc_df['full_text'].apply(vectorize_doc)], axis=1)


### PLOTTING ###
# # dimensionality reduction using t-sne
tsne = sklearn.manifold.TSNE(n_components=2, random_state=seed)


# fit tsne
word_embeddings_2d = tsne.fit_transform(doc_df[range(0, num_features)])

# put everything into a dataframe
doc_df['x'] = word_embeddings_2d[:,0] 
doc_df['y'] = word_embeddings_2d[:,1] 


# scatterplot
ax = doc_df.plot.scatter('x', 'y', s=10, figsize=(20,12))

# put word labels on scatterplot
for _, row in doc_df.iterrows(): 
    ax.annotate(row['title'], (row['x'], row['y']), size=10)
