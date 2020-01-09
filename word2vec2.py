import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
import nltk.data
import logging



pub_d = {
    'New York Times': 'NYT',
    'Breitbart': 'BB',
    'CNN': 'CNN',
    'Atlantic': 'ATL',
#    'Fox News': 'FOX',
#    'Talking Points Memo': 'TPM',
#    'Buzzfeed News': 'BUZ',
#    'National Review': 'NAT',
#    'New York Post': 'NYP',
#    'Guardian': 'GDN',
#    'NPR': 'NPR',
#    'Reuters': 'REU',
#    'Vox': 'VOX',
#    'Washington Post': 'WAPO'
#    'Business Insider': 'BI'
}

# shrink
#train = train.loc[train['publication']=='Business Insider']

remove_stopwords = True 
def content_to_wordlist( content, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    content_text = BeautifulSoup(content).get_text()
    #  
    # 2. Remove non-letters
    content_text = re.sub("[^a-zA-Z0-9' -]"," ", content_text)
    #
    # 3. Convert words to lower case and split them
    words = content_text.lower().split()
    #
    # 3.5 WordNetLemmatize
    lem = WordNetLemmatizer()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [lem.lemmatize(w) for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a content into parsed sentences
def content_to_sentences( content, tokenizer, remove_stopwords=False ):
    # Function to split a content into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(content.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call content_to_wordlist to get a list of words
            sentences.append( content_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # get data
    df = pd.read_csv('df_raw.csv')
    print('df loaded!')

    for pub in pub_d.keys():
        model = None;

        # select pub
        train = df.loc[df['publication']==pub]
        print("training on: ", pub)

        sentences = []  # Initialize an empty list of sentences
        print("Parsing sentences from training set")
        for content in train["content"]:
        #for content in df["content"]:
            sentences += content_to_sentences(content, tokenizer)
        
        # Set values for various parameters
        num_features = 300    # Word vector dimensionality                      
        min_word_count = 20   # Minimum word count                    
        num_workers = 4       # Number of threads to run in parallel
        context = 5           # Context window size                                                                                    
        downsampling = 1e-3   # Downsample setting for frequent words
        
        # Initialize and train the model (this will take some time)
        print("Training model...")
        model = Word2Vec(sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)
        
        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        #model.init_sims(replace=True)
        
        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        STORE_PATH = '/home/jtoma/s1/patternRecognition/project1/models'
        file_name = '300f_50mw_5c_' + pub_d[pub]
        #file_name = '300f_50mw_10c_COR'
        model_name = os.path.join(STORE_PATH, file_name)
        model.save(model_name)
        print("saving to:", model_name)

#model.doesnt_match("man woman child kitchen".split())
#model.doesnt_match("france england germany berlin".split())
#model.doesnt_match("paris berlin london austria".split())
#print('similar to "president":',model.most_similar("president"))
#print('similar to "trump":',model.most_similar("trump"))
#print('similar to "obama":',model.most_similar("obama"))
#print('similar to "election":',model.most_similar("election"))
#print('similar to "republican":',model.most_similar("republican"))
