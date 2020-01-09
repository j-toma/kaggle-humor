import pandas as pd


def get_data(articles):
    df1 = pd.read_csv(articles[0])
    df2 = pd.read_csv(articles[1])
    df3 = pd.read_csv(articles[2])
    df_raw = pd.concat([df1,df2,df3])
    return df_raw

def word_count(df):
    # by publication
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split(" ")))
    wc = df.groupby('publication').word_count
    pub_counts = wc.describe().to_dict()
    pub_counts["pub_wc"] = wc.sum().to_dict()
    # just keep article count, mean word count, and total word count
    keys_to_delete = ['sts','min','max','25%','50%','75%']
    for k in keys_to_delete:
        if k in pub_counts:
            del pub_counts[k]
    reorganized = {}
    for i in pub_counts['count'].keys():
        reorganized[i] = {'article_count':pub_counts['count'][i],
                          'mean_word_count':pub_counts['mean'][i],
                          'pub_word_count':pub_counts['pub_wc'][i]}
    # add total word count
    total_wc = df.word_count
    corpus = {'article_count':total_wc.describe().to_dict()['count'],
              'mean_word_count':total_wc.describe().to_dict()['mean'],
              'pub_word_count':total_wc.sum()}
    reorganized.update({'corpus':corpus})

    # convert to dataframe
    data = pd.DataFrame(reorganized).T
    #data.index.name = 'publication'

    return data


