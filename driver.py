import os

## IO Stuff ##
#json 
import json

# file pathnames
import glob

# regex
import re

# natural language toolkit
import nltk

# gensim for models
import gensim

# dimensionality reduction algorithms
import sklearn.manifold

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# pandas
import pandas as pd

# command line argument parser
import argparse

# custom utils function to make my life easier
import io_utils as io
import text_utils as text
import pdb


IMPORTANT_SECTIONS = ['body_text', 'abstract']

def init_kv_function(kv_dict, additional_kwargs=None):
    '''Initializes a `kv_function` 
    input: 
        kv_dict: [dict] Key-value pair for defining a function
        additional_kwargs: [dict] Additional kwargs to update `v` of kv_function
    '''
    if len(kv_dict) > 1:
        raise RuntimeError('a kv_dict should only contain 1 valid key & value (kwargs)')

    # check if we can import function
    function_str = list(kv_dict.keys())[0]
    function = eval(function_str)

    # check if kwargs are correct
    kwargs = kv_dict[function_str]
    if additional_kwargs is not None:
        kwargs.update(additional_kwargs)
    ret = function(**kwargs)

    return ret



# parse command-line arguments
p = argparse.ArgumentParser()
p.add_argument('--cfg', type=str, required=True)
args = vars(p.parse_args())

# store config file into a python dictionary
cfg = io.load_yaml(args['cfg'])
rseed = cfg['rseed']


#### Download Dependencies ####
## Download nltk dependencies ##
text.download_tokenizers(['punkt', 'stopwords'])

# load punkt tokenizer.  will be used to split sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


##### Input/Output Directory Setup #####
# 1. store all json filepaths from the CORD-19 dataset
# 2. create output directory structure:
#       base_output_directory
#       |       |
#       |---- model
#       |       |---- covid2vec.w2v
#       |---- json
#       |       |---- all json files from papers
#       |---- meta.csv


# CORD19 data directory 
DS_DIR = cfg['CORD19_dir']

# generate a list of all json filepaths from the CORD19 corpus
json_filepaths = glob.glob(os.path.join(DS_DIR, '*', '*', '*.json'))

# setup output directory structure
output_dir = cfg['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(os.path.join(output_dir, 'models'))
    os.makedirs(os.path.join(output_dir, 'json'))


### Preprocess CORD19 JSON files ###
# 1. read meta and json files
# 2. create a dataframe (corpus_df) to store parsed article data
# 3. save article metadata into corpus_df
#   - sha
#   - title
# 4. save important article text into corpus_df
#   - break sentences up into lists of tokens (words)
#   - store sentence tokens into said dataframe

# corpus df stores article data from the entire corpus
# each index will correspond to a single article.
# eg. corpus_df[title][0] will be the title of article 0
#     corpus_df[<section>][0] will be a list of tokenized sentences for article 0

meta_df = pd.read_csv(cfg['meta_path']).set_index('sha')
meta2keep = cfg['meta2keep']
corpus_df = {
    'title': [], # list of article titles
    'sha': []    # list of article ids
}

# create list of tokenized sentences from important sections
for section in IMPORTANT_SECTIONS:
    corpus_df[section] = []

# create list to keep track of metadata elements
for meta in meta2keep:
    corpus_df[meta] = []


# parse articles, and store text data in corpus_df
for i, filepath in enumerate(json_filepaths):

    # print progress so i don't lose my mind
    if (len(json_filepaths)/(i + 1)) % 100 == 0:
        print('Working on %i/%i' % (i + 1, len(json_filepaths)))

    # read article data from json
    article_data = io.load_json(filepath)

    # parse article metadata (sha, title)
    corpus_df['title'].append(article_data['metadata']['title'])
    sha = article_data['paper_id']
    corpus_df['sha'].append(sha)

    # save metadata elements if sha exists in the metadata csv
    try: 
        sha_meta = meta_df.loc[sha]
        for meta in meta2keep:
            corpus_df[meta].append(sha_meta[meta])
    except KeyError:
        for meta in meta2keep:
            corpus_df[meta].append(None)

    # parse important article text, tokenize, and store in corpus_df 
    # (currently only abstract and body_text)
    for section in ['abstract', 'body_text']:
        section_text = text.get_section_text(article_data[section])
        section_ps = []

        # tokenize sentences, and append to corpus_df
        for sentence in tokenizer.tokenize(section_text):
            tokenized_sent = text.sent2wordlist(sentence)
            if len(tokenized_sent) > 0:
                section_ps.append(tokenized_sent)
        corpus_df[section].append(section_ps)


# convert corpus_df into pandas dataframe for easy processing
corpus_df = pd.DataFrame(corpus_df)

# combine tokenized sentences from all important sections
# this will be used to train the w2v model
corpus_df['full_text'] = corpus_df[IMPORTANT_SECTIONS].sum(axis=1)

full_tokenized_corpus = [sent for article in corpus_df['full_text'].values for sent in article]

print('Dataset contains %i tokens.' % (sum([len(full_tokenized_corpus) for sentence in full_tokenized_corpus])))


#### Generate article embeddings ####
# 1. Train a word embedding model 
# 2. Use model to generate word-embeddings
# 3. Average word-embeddings over each article to generate article embeddings
# 4. Run t-SNE over article embeddings to retrieve relative coordinates for each article
covid2vec = init_kv_function(cfg['model'], additional_kwargs={'seed': rseed})

# build vocabulary
covid2vec.build_vocab(full_tokenized_corpus)
print('Vocab Length: %i' % len(covid2vec.wv.vocab))

# train w2v model
covid2vec.train(full_tokenized_corpus, total_examples=covid2vec.corpus_count, epochs=covid2vec.epochs)

# save model
covid2vec.save(os.path.join(output_dir, 'models','covid2vec.w2v'))

# store word embeddings into a dataframe.  
# it will be used as a lookup table aggregatings all words within a sentence
word_embedding_df = pd.DataFrame(covid2vec.wv.vectors)
word_embedding_df['word'] = covid2vec.wv.vocab
word_embedding_df = word_embedding_df.set_index('word')

def vectorize_doc(doc):
    ''' To be used with panda's apply function.
    Creates document embeddings by averaging word embeddings within the document

    # doc: list of tokenized sentences inside a document
    '''
    words = [sent for pas in doc for sent in pas]
    return word_embedding_df.reindex(words).mean()

# concatenate full document embedding into corpus_df
corpus_df = pd.concat([corpus_df, corpus_df['full_text'].apply(vectorize_doc)], axis=1)

print('Creating t-SNE embeddings')
# create t-SNE model to project document embedding on 2D for plot axes
tsne = sklearn.manifold.TSNE(n_components=2, random_state=rseed)

# fit tsne, and apply it on the document embeddings to get 2D projections
model_name = list(cfg['model'].keys())[0]
num_features = cfg['model'][model_name]['size']

word_embeddings_2d = tsne.fit_transform(corpus_df[range(0, num_features)])

# store 2D projects of document embedding in corpus_df
corpus_df['x'] = word_embeddings_2d[:,0]
corpus_df['y'] = word_embeddings_2d[:,1]


#### Save files ####
# JSON
corpus_df[IMPORTANT_SECTIONS + meta2keep + ['x', 'y']].to_json(os.path.join(output_dir, 'articles.json'), 
                                                               orient='records', lines=True)

# hdf
corpus_df.to_hdf(os.path.join(output_dir, 'corpus_df.hdf'), 'data')