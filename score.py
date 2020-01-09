from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math
import operator

def score(y,y_pred,scaler,ensemble=False):
    if ensemble:
        y = y.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)

    # revert to 0-3 scale and print total score
    y = scaler.inverse_transform(y)
    y_pred = scaler.inverse_transform(y_pred)
    mse = mean_squared_error(y, y_pred)
    print('total rmse:',math.sqrt(mse))
    total_rmse = round(math.sqrt(mse),4)

    # get bucket performance
    pairs = [(y[i],y_pred[i]) for i in range(len(y))]
    sorted_by_y = pairs.sort(key=operator.itemgetter(0))
    l = len(y)
    bucket_sizes = [10,20,30,40]
    buckets = []
    for N in bucket_sizes:
        bottom = pairs[:int(l*N/100)]
        top = pairs[int(l*(100-N)/100):]
        bkt = bottom + top
        bkt_y = list(map(operator.itemgetter(0),bkt))
        bkt_y_pred = list(map(operator.itemgetter(1),bkt))
        rmse = math.sqrt(mean_squared_error(bkt_y, bkt_y_pred))
        bucket = 'N: {} -- rmse: {}'.format(N, round(rmse,4))
        buckets.append(bucket)
    return total_rmse, buckets, y, y_pred

