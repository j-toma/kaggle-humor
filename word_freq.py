from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd
import seaborn as sns

pub_d = {
    'New York Times': 'NYT',
    'Breitbart': 'BB',
    'CNN': 'CNN',
    'Atlantic': 'ATL',
    'Fox News': 'FOX',
    'Talking Points Memo': 'TPM',
    'Buzzfeed News': 'BUZ',
    'National Review': 'NAT',
    'New York Post': 'NYP',
    'Guardian': 'GDN',
    'NPR': 'NPR',
    'Reuters': 'REU',
    'Vox': 'VOX',
    'Washington Post': 'WAPO',
    'Business Insider': 'BI'
}

stops = set(stopwords.words('english'))

def clean(content):
    content = str(content)

    #content = BeautifulSoup(content).get_text()

    #Remove punctuations
    text = re.sub("[^a-zA-Z0-9]", ' ', content)
    
    #Convert to lowercase
    words = text.lower().split()
    
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in words if word not in  
            stops] 
    text = " ".join(text)
    return text

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqDict):
    aux = sorted(freqDict.items(), key=lambda item: item[1])
    aux.reverse()
    return aux[:20]


if __name__ == '__main__':
    
    #stop_words_col = dict()
    #data = pd.read_csv('df_raw.csv', chunksize=1000)

    # full corpus counts
    #print('processing corpus')
    #top_words_corpus = dict()
    #count = 0
    #for chunk in data:
    #    print('processing chunk number:', count)
    #    chunk['content'] = chunk['content'].map(clean)
    #    word_freq = chunk.content.str.split(expand=True).stack().value_counts().to_dict()
    #    for w in word_freq.keys():
    #        if w in top_words_corpus:
    #            top_words_corpus[w] = top_words_corpus[w] + word_freq[w]
    #        else:
    #            top_words_corpus[w] = word_freq[w]
    #    count += 1
    #top_words = sorted(word_freq.items(), key = lambda x:x[1], reverse=True)[:20]
    #top_words_col.update({'corpus': top_words})
    #print('corpus processing finished:')
    #print('corpus top words:', top_words)

    # by pub counts
    #for pub in pub_d.keys():
    #    data = pd.read_csv('df_raw.csv')
    #    print('WORKING ON:', pub)
    #    data = data.loc[data['publication']==pub]
    #    data = data['content'].map(clean)
    #    word_freq = data.str.split(expand=True).stack().value_counts().to_dict()
    #    top_words = sorted(word_freq.items(), key = lambda x:x[1], reverse=True)[:20]
    #    top_words_col.update({pub: top_words})
    #    print('finished processing top words for:',pub)
    #    print(top_words)
    #    print('-------------------------------')

    wc = {k:{} for k in pub_d.keys()}
    wc['corpus'] = {}
    data = pd.read_csv('df_raw.csv')
    count = 0 
    for index,row in data.iterrows():
        if count%1000==0:
            print('WORKING ON ROW:', count)
        content = clean(row['content'])
        pub = row['publication']
        word_freq = wordListToFreqDict(content.split())
        for w in word_freq.keys():
            if w in wc['corpus']:
                wc['corpus'][w] = wc['corpus'][w] + word_freq[w]
            else:
                wc['corpus'][w] = word_freq[w]
            if w in wc[pub]:
                wc[pub][w] = wc[pub][w] + word_freq[w]
            else:
                wc[pub][w] = word_freq[w]
        count+=1
    print('loop finished')
    top_words = dict() 
    for k in wc.keys():
        top_words[k] = sortFreqDict(wc[k]) 
    print(top_words)

    stats = pd.read_csv('stats.csv')
    stats['top_words'] = stats['publication'].map(top_words)
    stats.to_csv('stats.csv', index=False)

    print(stats)

#Barplot of most freq words
#sns.set(rc={'figure.figsize':(13,8)})
#g = sns.barplot(x="Word", y="Freq", data=top_df)
#g.set_xticklabels(g.get_xticklabels(), rotation=30)
