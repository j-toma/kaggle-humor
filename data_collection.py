#import csv
#import sys
#import re
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer
#from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
from get_data import get_data
from word_count import word_count


def main():
    # get data
    try:
        df_raw = pd.read_csv('df_raw.csv')
    except IOError:
        articles = ['articles1.csv', 'articles2.csv', 'articles3.csv']
        df_raw = get_data(articles)
        df_raw.to_csv('df_raw.csv')

    # word count and df initialization 
    try:
        stats_by_pub = pd.read_csv('stats_by_pub.csv')
        print(stats_by_pub)
        print('stats on disk')
    except IOError:
        stats_by_pub = word_count(df_raw)
        stats_by_pub.to_csv('stats_by_pub.csv')


if __name__ == "__main__":
    main()

#def plot_publications(d):
#    publications = tuple(d.keys())
#    y_pos = np.arange(len(d.keys()))
#    no_articles = d.values()
#    plt.barh(y_pos, no_articles, align='center', alpha=0.5)
#    plt.yticks(y_pos, publications)
#    plt.xlabel('Number of articles')
#    plt.title('News Sources')
#    
#    plt.show()

## ------------------------- CODE GRAVEYARD ----------------------------- ##

# avoid fieldsize issue
#csv.field_size_limit(sys.maxsize)

## removed from main as function it calls is unnecessary 
    ## agglomerate content 
    #try:
    #    content_by_pub = pd.read_csv('content_by_pub.csv')
    #    print('content on disk')
    #except IOError:
    #    content_by_pub= agglomerate(df_raw)
    #    content_by_pub.to_csv('content_by_pub.csv')

# moved to separate file
#def word_count(df):
#    # by publication
#    df['word_count'] = df['content'].apply(lambda x: len(str(x).split(" ")))
#    wc = df.groupby('publication').word_count
#    pub_counts = wc.describe().to_dict()
#    pub_counts["pub_wc"] = wc.sum().to_dict()
#    # just keep article count, mean word count, and total word count
#    keys_to_delete = ['sts','min','max','25%','50%','75%']
#    for k in keys_to_delete:
#        if k in pub_counts:
#            del pub_counts[k]
#    reorganized = {}
#    for i in pub_counts['count'].keys():
#        reorganized[i] = {'article_count':pub_counts['count'][i],
#                          'mean_word_count':pub_counts['mean'][i],
#                          'pub_word_count':pub_counts['pub_wc'][i]}
#    # add total word count
#    total_wc = df.word_count
#    corpus = {'article_count':total_wc.describe().to_dict()['count'],
#              'mean_word_count':total_wc.describe().to_dict()['mean'],
#              'pub_word_count':total_wc.sum()}
#    reorganized.update({'corpus':corpus})
#
#    # convert to dataframe
#    data = pd.DataFrame(reorganized).T
#    #data.index.name = 'publication'
#
#    return data  

# moved to separate file
#def get_data():
#    articles = ['articles1.csv', 'articles2.csv', 'articles3.csv']
#    df1 = pd.read_csv(articles[0])
#    df2 = pd.read_csv(articles[1])
#    df3 = pd.read_csv(articles[2])
#    df_raw = pd.concat([df1,df2,df3])
#    return df_raw

# get list of publications
# do not need this. selective preprocessor is fine
#def agglomerate(df_raw):
#    content_column = df_raw.groupby('publication')['content'].agg(lambda col: ' '.join(col))
#    total_content = df_raw['content'].agg(lambda col: ' '.join(col))
#    #total_content = content_column.agg(lambda col: ' '.join(col))
#    content_column.append(total_content)
#    # convert to df
#    data = pd.DataFrame({'publication':content_column.index,
#                         'content':content_column.values})
#    return data 


    #print('sep shape:',df_sep.shape)
    #print(df_sep)
    #print('cont col shape:',content_column.shape)
    #print(content_column)
    #### freezes computer for a minute then prints 'Killed' ;)
    #df_sep.update(content_column)

    #### no memory error but is taking forever
    # get agglomerated content df
    #pubs = list(df_sep.index)
    #for row in df_raw.itertuples(index=True, name='Pandas'):
    #    pub = getattr(row,'publication')
    #    content = getattr(row,'content')
    #    df_sep.at[pub,'content'] += content

    #### Memory error
    #contents = {}
    #total_content = ''
    #for pub in pubs: 
    #    content = ' '.join(df_raw.loc[df_raw['publication']==pub]['content']) 
    #    contents[pub] = content
    #    total_content += content
    #contents['corpus'] = total_content
    #df_agglomerated_content_column = pd.DataFrame(contents,dtype='object')
    # merge 
    #print('sep shape:',df_sep.shape)
    #print(df_sep)
    #print('agglo shape:', df_agglomerated_content_column.shape)
    #df = pd.concat([df_sep,df_agglomerated_content_column],axis=1)
    #return df

#def publications(articles):
#    pubs = [] 
#    pubs_counts = {}
#    pubs_preproc_content = {}
#    for article in articles:
#        f = open(article)
#        df = pd.read_csv(f)
#        # set of publications 
#        pubs = pubs + list(df.publication.unique())
#
#        # count of articles for each publication (not necessary anymore -- see
#        # pubs_counts.
#        # update(df.groupby('publication')['id'].nunique().to_dict())
#
#        # word counts
#        pubs_counts.update(word_count(df))
#
#        # preprocess for individual publications
#        for pub in pubs:
#            preprocess(df.loc[df['publication'] == pub])
#        f.close()
#    return pubs,pubs_counts
#
#
#def preprocess(df):
#    stop_words = set(stopwords.words("english"))
#    corpus = []
#    for i in range(df.size):
#        # remove punctuation
#        text = re.sub('[^a-zA-Z]', ' ', df['content'][i])
#
#        # convert to lower
#        text = text.lower()
#
#        # remove special characters and digits
#        text = re.sub("(\\d|\\W)+"," ",text)
#
#        # lemmatize
#        lem = WordNetLemmatizer()
#        text = [lem.lemmatize(word) for word in text if not word in stop_words]
#        text = " ".join(text)
#        corpus.append(text)
#        
#
## get all sources and their counts
##def publications(articles):
##    pubs = dict()
##    for article in articles:
##        f = open(article)
##        d = csv.DictReader(f)
##        for row in d:
##            if row['publication'] not in pubs:
##                pubs[row['publication']] = 0
##            else:
##                pubs[row['publication']] += 1
##        f.close()
##    return pubs
#
#
#pubs,pubs_counts = publications(articles)
#print(pubs)
#print(pubs_counts)
##plot_publications(pubs)
#
## text files for all content from each source
#def separate():
#    for article in articles:
#        f = open(article)
#        d = csv.DictReader(f)
#        total_content = open('content/total_content.txt','a')
#        for row in d:
#            publication = row['publication']
#            content = open('content/%s.txt' % publication,'a')
#            content = row['content']
#            content.write(row)
#            total_content.write(content)
#            content.close()
#        total_content.close()
#        f.close()
#
## clean 
#def preprocess():
#    for pub in pubs:
#        f = open('content/%s.txt' % pub, 'r')
#        data = f.read()
#        f.close()
#
#        stop_words = set(stopwords.words("english"))
#        new_words = [str(pub) for pub in pubs]
#        stop_words = stop_words.union(new_words)
#
#
#        f = open('preprocessed/%s.txt' % pub)
#        f.write(data)
#        f.close()
#
