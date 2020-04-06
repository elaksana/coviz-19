from . import io_utils
import pandas as pd
import glob


# class Paper:
#     def __init__(self, json_filepath, **kwargs):
#         ''' Class to easily vectorize document and access attributes 
#         '''

#         # read json data
#         json_data = io_utils.load_json(json_filepath)
        
#         ## STORE ATTRIBUTES ##
#         # document text
#         def _get_section_text(section_data):
#             ''' concatenates text data from section data (extracted from json)
#             inputs:
#                 section_data[dict]: dictionary representation of text section from the CORD-19 corpus
#             '''
#             text = u''
#             # concatenate text for each paragraph
#             for p_data in section_data:
#                 text += paragraph_data['text']

#         self.abstract = _get_section_text(json_data['abstract'])
#         self.body = _get_section_text(json_data['body'])

#         # store other keywords passed through the paper class to be stored
#         self.__dict__.update(kwargs)

#     def save_paper(self, ofilepath, do_not_save = ['abstract', 'body']):
#         ''' Save papers as json to ofilepath
#         inputs:
#             ofilepath [str]: path to save json file
#             do_not_save[list(str)]: list of attributes to not save
#         '''
        
#         import pdb
#         pdb.set_trace()
#         pass

class Corpus:

    def __init__(self, cfg, parser, **kwargs):
        ''' Class to easily access and perform operations on all documents in the CORD-19 corpus
        '''

        # get list of json filepaths`
        json_filepaths = glob.glob(os.path.join(cfg['CORD19_dir'], '*', '*', '*.json'))

        # number of documents in corpus
        self.n_docs = len(self.json_filepaths)

        # document list iterator
        # self.doc_iter = 0

        # read metadata, and index by sha (doc id and also filename)
        self.meta_df = pd.read_csv(cfg['meta_path']).set_index('sha')
        self.meta2keep = cfg['meta2keep']

        self.parser = parser


        self.doc_df = {
            'tokenized_text': [],
            'sha': []
        }
        
        def _get_section_text(section_data):
                ''' concatenates text data from section data (extracted from json)
                    inputs:
                        section_data[dict]: dictionary representation of text section from the CORD-19 corpus
                '''
                text = u''
                # concatenate text for each paragraph
                for p_data in section_data:
                    text += paragraph_data['text']


        for filepath in json_filepaths:
            # load json data
            json_data = io_utils.load_json(filepath)
            sha = json_data['paper_id']

            # extract relevant info from json file
            title = json_data['metadata']['title']
            abstract = _get_section_text(json_data['abstract'])
            body = _get_section_text(json_data['body_text'])

            # tokenize abstract and body
            tokenized_paragraphs = self.parser.tokenize_passage(abstract + body)

            self.doc_df['tokenized_text'].append(tokenized_paragraphs)
            self.doc_df['sha'].append(sha)
            self.doc_df['title'].append(title)

        self.doc_df = pd.DataFrame(doc_df)


    def tokenize(self):
        ''' 
            Create a list of list of words... treats the entire corpus as one very long passage with tons of sentence tokens

            input: nada
            output: 
                tokenized_corpus [list(list(str))]:
                    list of list of sentences in the corpus to be used to train w2v model

        '''
        return [sent for doc in self.doc_df['tokenized_text'].values for sent in doc]


    def embed2d(self, trained_model):
        ''' Create vectorized representation of documents using embeddings from trained_model 

        input: 
            trained_model [gensim.models.word2vec]: fitted word2vec model from the gensim library
        output:
            doc_vec [np.array]: vectorized representation of document
        '''
        # ## DOCUMENT-LEVEL EMBEDDING ##
        # 1. store word embeddings into pandas dataframe for simple lookup
        # 2. get intradocument word vector average
        word_embedding_matrix = trained_model.wv.vectors
        word_embedding_df = pd.DataFrame(word_embedding_matrix)
        word_embedding_df['word'] = trained_model.wv.vocab
        word_embedding_df = word_embedding_df.set_index('word')
        pass

    def __len__(self):
        # return the number of papers in corpus
        return self.n_docs


    # def __next__(self): 
        # returns the next paper object
        # if self.doc_iter < self.n_docs
        #     json_filepath = self.json_filepaths[self.doc_iter]
        #     base = os.path.basename(json_filepath)
        #     doc_id = os.path.splitext(base)[0] 

        #     # instantiate paper object
        #     try: 
        #         # instantiate paper with meta attributes if it can be found in meta_df
        #         kwarg_dict = meta_df.loc[doc_id, self.meta2keep].to_dict()
        #     except KeyError
        #         # instantiate paper with null meta attributes otherwise
        #         kwarg_dict = dict.fromkeys(self.meta2keep)

        #     p = Paper(json_filepath, **kwarg_dict)
            
        #     # increment document iterator
        #     self.doc_iter += 1

        #     return p
        
        else: 
            raise StopIteration
    
    def __iter__(self):
        # reset document iterator
        self.doc_iter = 0
        return self
