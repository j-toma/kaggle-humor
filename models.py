from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D
from keras.layers import Activation, MaxPooling1D, GRU
from keras.layers import Embedding, SpatialDropout1D, Flatten 
from keras.layers import CuDNNGRU, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from Attention_layer import AttentionM



class Models:
    def __init__(self, input_shape, maxlen, model_types, hparams, weights=None):
        self.input_shape = input_shape
        self.maxlen = maxlen
        self.models = model_types 
        self.nb_filter = hparams['nb_filter']
        self.kernel_size = hparams['kernel_size']
        self.hidden_dims = hparams['hidden_dims']
        self.vector_size = hparams['vector_size']
        self.pool_size = hparams['pool_size']
        self.lstm_output_size = hparams['lstm_output_size']
        self.weights = weights 
        self.max_features = self.weights.shape[0]
        self.num_features = self.weights.shape[1]


    def lstm(self):
        model = Sequential()
        model.add(Embedding(
            input_dim=self.max_features,
            output_dim=self.num_features,
            input_length=self.maxlen,
            #mask_zero=True,
            weights=[self.weights],
            trainable=False
        ))
        model.add(Dropout(0.2))
        model.add(LSTM(self.hidden_dims, recurrent_dropout=0.2, return_sequences=True))
        #model.add(Dropout(0.2))
        #model.add(LSTM(self.hidden_dims, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(AttentionM())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mse',optimizer='adam')

        return model


    def cnn(self):
        model = Sequential()
        model.add(Embedding(
            input_dim=self.max_features,
            output_dim=self.num_features,
            input_length=self.maxlen,
            #mask_zero=True,
            weights=[self.weights],
            trainable=False
        ))
        model.add(Dropout(0.3))
        model.add(Conv1D(self.nb_filter,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))

        #model.add(Dropout(0.2))
        model.add(GlobalMaxPooling1D())
        #model.add(MaxPooling1D(pool_size=self.pool_size))
        #model.add(AttentionM())
        #model.add(Dropout(0.25))

        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='mse',optimizer='adam')

        return model


    def gru(self):
        model = Sequential()
        model.add(Embedding(
            input_dim=self.max_features,
            output_dim=self.num_features,
            input_length=self.maxlen,
            #mask_zero=True,
            weights=[self.weights],
            trainable=False
        ))
        model.add(Dropout(0.5))
        model.add(GRU(self.hidden_dims // 2, recurrent_dropout=0.25, return_sequences=True))
        model.add(AttentionM())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')

        return model


