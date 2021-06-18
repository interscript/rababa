
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


import torchinfo
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from pipeline_diacritizer.dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, clear_diacritics, add_time_steps, NUMBER_REGEXP, WORD_TOKENIZATION_REGEXP, ZERO_REGEXP, ARABIC_LETTERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
    
    We implement the model of Fig.3 of 
    docs/2020-multi-components-system-for-automatic-arabic-diacritization.pdf, 
    replicating the code of https://github.com/Hamza5/Pipeline-diacritizer.
    
    Use the code commented below to output models size and generate an architecture comparison.

"""


class PipelineModel(torch.nn.Module):
    
  def __init__(self,
               nb_input_features, nb_timesteps,
               h1_features=64,
               num_lstm_layers=2):
    super().__init__()
    
    self.nb_input_features = nb_input_features
    self.nb_timesteps = nb_timesteps
    
    # multilayered LSTM
    self.lstm_2l = torch.nn.LSTM(input_size=nb_input_features,
                                 hidden_size=h1_features,
                                 dropout=0.1,
                                 bidirectional=True,
                                 num_layers=num_lstm_layers)
    
    # flatten
    self.flat = torch.nn.Flatten()
    
    # dims after flattening
    self.inner_features_ = nb_timesteps * h1_features * 2 # * 2 because bidirectional
    
    self.tanh_act = nn.Tanh()
    self.sigmoid_act = nn.Sigmoid()
    self.softmax_act = nn.Softmax()
    
    # Shadda "branch"
    self.nb_features_64, self.nb_features_8, self.nb_features_1 = 64, 8, 1
    self.dense_shadda_8 = nn.Linear(self.inner_features_, self.nb_features_8, bias=True)
    self.dense_shadda_1 = nn.Linear(self.nb_features_8, self.nb_features_1, bias=True)
    
    # Harakat "branch"
    self.dense_haraka_64 = nn.Linear(self.inner_features_, self.nb_features_64, bias=True)
    self.dense_haraka_8 = nn.Linear(self.nb_features_64, self.nb_features_8, bias=True)


  def forward(self, x):
    
    # lstm + flattening
    x, (h1_T,c1_T) = self.lstm_2l(x)
    x = self.flat(x)
    
    # shadda branch
    x_shadda = self.tanh_act(self.dense_shadda_8(x))
    x_shadda = self.sigmoid_act(self.dense_shadda_1(x_shadda))
    
    # harakat branch
    x_haraka = self.tanh_act(self.dense_haraka_64(x))
    x_haraka = self.softmax_act(self.dense_haraka_8(x_haraka))
    
    return  x_shadda, x_haraka


  def hot_to_categorical(self, indices, n_char2idx):
    """ Helper hot_to_categorical:
            util transforming hot vector into categories: 
                                [[0,0,1,0,0], ...] --> [2, ]
        param: indices hot indices
    """
    
    n_idces = len(indices)
    tensor = np.zeros((n_idces, n_char2idx))
    for i in range(n_idces):
        tensor[i][indices[i]] = 1
        
    return tensor


  def text_indices_2_data(self, text_indices, diacritics_indices, idx):
    """ Helper:
            preprocess data into hot data and target data
        param: text_indices vecrtor representation of text
        param: diacritics_indices mapping to arabic text representation
        param: idx index to get data row
    """
    
    hot_data = add_time_steps(self.hot_to_categorical(text_indices[idx], 38),
                              self.nb_timesteps, False)
    target_harakat = self.hot_to_categorical(diacritics_indices[idx][1], self.nb_features_8)
    target_shadda = np.array(diacritics_indices[idx][0], dtype=np.float).reshape((-1, 1))
    
    return hot_data, [target_shadda, target_harakat]


""" Utils
"""

def get_idx_harakat(target):
    """ Helper:
            hot to integer for harakat
    """
    for i in range(target.shape[0]):
        if target[i]==1.:
            return i
    return 0


""" Metrics
"""

def precision_pt(target, predict):
    true_positives = sum([np.dot(predict[i], target[i]) for i in range(predict.shape[0])])
    predicted_positives = sum([np.dot(predict[i], predict[i]) for i in range(predict.shape[0])])
    precision = true_positives / (predicted_positives + np.finfo(np.float32).eps) # + K.epsilon())
    return precision


def recall_pt(target, predict):
    true_positives = sum([np.dot(predict[i], target[i]) for i in range(predict.shape[0])])
    possible_positives = sum([np.dot(target[i], target[i]) for i in range(target.shape[0])])
    recall = true_positives / (possible_positives + np.finfo(np.float32).eps) # + K.epsilon())
    return recall


def binary_accuracy_pt(target, predict):
    predict = np.array(list(map(lambda d: int(d>=0.5), predict)))
    target = np.array(list(map(lambda d: int(d), target)))
    return accuracy_score(predict, target)


def categorical_accuracy_pt(target, predict, thresh=0.5):
    predict = np.array(list(map(lambda d: np.argmax(d), predict)))
    return accuracy_score(predict, target)





"""
***********************************************************************
"""
#from dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, \
#    clear_diacritics, add_time_steps, NUMBER_REGEXP, WORD_TOKENIZATION_REGEXP, ZERO_REGEXP, ARABIC_LETTERS


def to_categorical_pt(indices, n_char2idx): #len(CHAR2INDEX)):
    
    n_idces = len(indices)
    tensor = np.zeros((n_idces, n_char2idx))
    for i in range(n_idces):
        tensor[i][indices[i]] = 1
        
    return tensor

def text_indices_2_data(text_indices, diacritics_indices, index):

    input = add_time_steps(to_categorical_pt(text_indices[index], 38), #len(CHAR2INDEX)),
                           10, False)
    
    target_harakat = to_categorical_pt(diacritics_indices[index][1], 8)
    target_shadda = np.array(diacritics_indices[index][0], dtype=np.float).reshape((-1, 1))
    
    return input, [target_shadda, target_harakat]

"""
***********************************************************************
"""



def train_model(nn_model, text_indices, diacritics_indices, nb_epochs=15):
    
    # stores values for monitoring
    precs_shadda, recalls_shadda = [], []
    precs_harakat, recalls_harakat = [], []
    losses, losses_harakat, losses_shadda = [], [], []
    bin_accurs_shadda, cat_accurs_harakat = [], []

    # Harakat "branch"
    loss_harakat = nn.CrossEntropyLoss() # weight=torch.Tensor(harakat_weights)) discarded for now
    optimizer_harakat = torch.optim.Adadelta(nn_model.parameters())

    # Shadda "branch"
    loss_shadda = nn.BCELoss() # weight=torch.Tensor([shadda_weight])) discarded for now
    optimizer_shadda = torch.optim.Adadelta(nn_model.parameters())

    for epoch in range(nb_epochs):
        
        for idx in tqdm.tqdm(range(len(text_indices))):
        
            # prepare data
            in_data, (target_shadda, target_harakat) = nn_model.text_indices_2_data(text_indices, diacritics_indices, idx)
            #in_data, (target_shadda, target_harakat) = text_indices_2_data(text_indices, diacritics_indices, idx)

        
            # predict through the network
            predict_shadda, predict_harakat = nn_model(torch.tensor(in_data).float())

            l_harakat = loss_harakat(predict_harakat, 
                                     torch.from_numpy(np.array(list(map(lambda x: get_idx_harakat(x), target_harakat)))))
            optimizer_harakat.zero_grad()
            l_harakat.backward(retain_graph=True)
            optimizer_harakat.step()
    
            l_shadda = loss_shadda(predict_shadda.flatten().float(),
                                   torch.from_numpy(target_shadda.flatten()).float())
            optimizer_shadda.zero_grad()
            l_shadda.backward()
            optimizer_shadda.step()
        
            l_harakat = float(l_harakat.detach().numpy())
            l_shadda = float(l_shadda.detach().numpy())
            l_total = l_harakat+l_shadda
        
            bin_accuracy = binary_accuracy_pt(target_shadda.flatten(),
                                              predict_shadda.flatten().detach().numpy())
            cat_accuracy = categorical_accuracy_pt(np.array(list(map(lambda d: np.argmax(d), target_harakat))),
                                                   predict_harakat.detach().numpy())
            bin_accurs_shadda.append(bin_accuracy)
            cat_accurs_harakat.append(cat_accuracy)
    
        
            losses_harakat.append(l_harakat)
            losses_shadda.append(l_shadda)
            losses.append(l_total)
            
            precs_harakat.append(precision_pt(target_harakat, predict_harakat.detach().numpy()))
            recalls_harakat.append(recall_pt(target_harakat, predict_harakat.detach().numpy()))
    
            precs_shadda.append(precision_pt(target_shadda, predict_shadda.detach().numpy()))
            recalls_shadda.append(recall_pt(target_shadda, predict_shadda.detach().numpy()))
    
            print('epoch:', epoch,
                  'loss:', sum(losses[-1000:])/len(losses[-1000:]), # l_harakat + l_shadda, 
                  'loss_harakat:', sum(losses_harakat[-1000:])/len(losses_harakat[-1000:]), 
                  'loss_shadda:', sum(losses_shadda[-1000:])/len(losses_shadda[-1000:]), 
                  'precision_harakat:', sum(precs_harakat[-1000:])/len(precs_harakat[-1000:]),
                  'precision_shadda:', sum(precs_shadda[-1000:])/len(precs_shadda[-1000:]),
                  'recall_harakat:', sum(recalls_harakat[-1000:])/len(recalls_harakat[-1000:]),
                  'recall_shadda:', sum(recalls_shadda[-1000:])/len(recalls_shadda[-1000:]), 
                  'bin_accuracy_shadda', sum(bin_accurs_shadda[-1000:])/len(bin_accurs_shadda[-1000:]),
                  'cat_accuracy_harakat', sum(cat_accurs_harakat[-1000:])/len(cat_accurs_harakat[-1000:]),
                  '\n')
    
    # write model for each epoch
    print('print model after epoch nb: ', epoch)
    torch.save(model, 'models/basis_model_n_epochs_'+str(epoch)+'.pkl')


"""

    Tensorflow original version:
    
    import numpy as np
    import tensorflow.keras.backend as K
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, TerminateOnNaN
    from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Input, Layer, Lambda
    from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
    from tensorflow.keras.optimizers import Adadelta
    from tensorflow.keras.utils import Sequence, to_categorical

    from dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, \
        clear_diacritics, add_time_steps, NUMBER_REGEXP, WORD_TOKENIZATION_REGEXP, ZERO_REGEXP, ARABIC_LETTERS


    input_layer = Input(shape=(TIME_STEPS, len(CHAR2INDEX)))

    inner_layers = [
            Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L1'),
            Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L2'),
            Flatten(name='F'),
            (Dense(8, activation='tanh', name='D1'), Dense(64, activation='tanh', name='D2'))
        ]
    previous_layer = input_layer
    for layer in inner_layers[:-1]:
        previous_layer = layer(previous_layer)

    shadda_side, haraka_side = inner_layers[-1]
    shadda_side = shadda_side(previous_layer)
    haraka_side = haraka_side(previous_layer)

    output_shadda_layer = Dense(1, activation='sigmoid', name='D3')(shadda_side)
    output_haraka_layer = Dense(8, activation='softmax', name='D4')(haraka_side)
    shadda_corrections_layer = Lambda(lambda x: x, name='output_shadda')(
                        [input_layer, output_shadda_layer, output_haraka_layer])

    haraka_corrections_layer = Lambda(lambda x: x, name='output_haraka')(
                        [input_layer, output_haraka_layer, output_shadda_layer])
    model = Model(inputs=input_layer,
                  outputs=[shadda_corrections_layer, haraka_corrections_layer])
    
    # ((
    # SHADDA:
    # inner_layers = [
    #        Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L1'),
    #        Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L2'),
    #        Flatten(name='F'),
    #        Dense(8, activation='tanh', name='D1'),
    #        Dense(1, activation='sigmoid', name='D3')]
    # HARAKA:
    # inner_layers = [
    #        Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L1'),
    #        Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L2'),
    #        Flatten(name='F'),
    #        Dense(64, activation='tanh', name='D2'),
    #        Dense(8, activation='softmax', name='D4')]
    # ))
    

    model.summary()

"""
