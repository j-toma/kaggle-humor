import os
import pickle
import numpy as np
from score import score
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    STORE_PATH = '/home/jtoma/s1/patternRecognition/project1/embeddings'
    file_name = 'ATL.pickle'
    pickle_file = os.path.join(STORE_PATH, file_name)
    data, vocab, maxlen, W, word_idx_sent = pickle.load(open(pickle_file, 'rb'))

    grades = []
    y = []
    for i in range(len(data)):
        if data[i]['split'] == 1:
            grades.append(data[i]['y'])
        if data[i]['split'] == 0:
            y.append(data[i]['y'])
        
    avg = sum(grades)/len(grades)
    avg = [avg for i in range(len(y))] 
    print(len(avg))
    
    scaler = MinMaxScaler()
    avg = np.array(avg).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    scaled_avg = scaler.fit_transform(avg)
    scaled_y = scaler.transform(y) 

    rmse, bkts, y, y_pred = score(y,avg,scaler)

