import os
import csv
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
FTRAIN = 'home/ubuntu/data/train_images_128_128_all_new.txt'
#FTEST = 'home/ubuntu/data/test_images_128_128_data.txt'
FTEST = "/home/ubuntu/data/test_images_128_128_all_new.txt"
"""Loads data from FTEST if *test* is True, 
otherwise from FTRAIN..
"""
os.chdir('../')
def load(test=False):
    def extractNumbers(R):
        return int(R.strip("'"))
    fname = FTEST if test else FTRAIN
    with open(fname, 'r') as f: # note text mode, not binary
        rows = (list(map(np.float32, row)) for row in csv.reader(f)) 
        df= np.vstack(rows)  
    X = df[:,range(1,df.shape[1])]  
    X = np.array(X).reshape( -1,1,128,128)
    # only FTRAIN has any target columns
    if not test: 
        y = df[:,0]
        y = y.astype(np.uint8).reshape(-1)
        X, y = shuffle(X, y, random_state=42) 
    else:
        y=None
    return X, y

import sys
import theano
def float32(k):
    return np.cast['float32'](k)

        
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)    
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

from lasagne import layers
import lasagne.nonlinearities
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax ,#None,
    #update_learning_rate=0.01,
    #update_momentum=0.9,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=214),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=300,
    verbose=1,
    )
###execute


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 128, 128),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden4_num_units=500, ####1000
    dropout4_p=0.5,
    hidden5_num_units=500, ####1000
    output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax,
    #update_learning_rate=0.01,
    #update_momentum=0.9,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    regression=False,
    batch_iterator_train=BatchIterator(batch_size=214),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=300,####10000
    verbose=1,
    )

import sys
sys.setrecursionlimit(10000)

X, y = load()    

import sys
sys.setrecursionlimit(10000)
X, _ = load(True)  

net1.fit(X, y)
# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('home/ubuntu/data/net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

net2.fit(X, y)
# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('home/ubuntu/data/net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)

import cPickle 
f = file('home/ubuntu/data/net1.pickle', 'rb')
loaded_obj = cPickle.load(f)
f.close()

y_pred = net2.predict(X)
import sklearn.metrics as metrics   
metrics.accuracy_score(y, y_pred)
0.91342102266256686

metrics.confusion_matrix(y, y_pred)

array([[25689,    21,    87,     8,     3],
       [  835,  1603,     5,     0,     0],
       [ 1665,     5,  3618,     3,     1],
       [  226,     0,    16,   631,     0],
       [  157,     1,     8,     0,   542]])

metrics.classification_report(y, y_pred)
'             precision    recall  f1-score   support\n\n
            0       0.90      1.00      0.94     25808\n          
            1       0.98      0.66      0.79      2443\n          
            2       0.97      0.68      0.80      5292\n          
            3       0.98      0.72      0.83       873\n          
            4       0.99      0.77      0.86       708\n
\navg / total       0.92      0.91      0.91     35124\n'


y_pred = net1.predict(X)
import sklearn.metrics as metrics   
metrics.accuracy_score(y, y_pred)
 0.93801958774627037
metrics.confusion_matrix(y, y_pred)
array([[25387,    89,   291,    22,    19],
       [  440,  1959,    38,     2,     4],
       [  932,    21,  4332,     3,     4],
       [  145,     8,    16,   699,     5],
       [  108,     2,    26,     2,   570]])
metrics.classification_report(y, y_pred)
'             precision    recall  f1-score   support\n\n
        0       0.94      0.98      0.96     25808\n          
        1       0.94      0.80      0.87      2443\n          
        2       0.92      0.82      0.87      5292\n          
        3       0.96      0.80      0.87       873\n          
        4       0.95      0.81      0.87       708\n
\navg / total       0.94      0.94      0.94     35124\n'



import sys
sys.setrecursionlimit(10000)

import cPickle as pickle
with open('home/ubuntu/data/net1.pickle', 'rb') as f:
    net1 = pickle.load( f )
    
def predictTest
    import os
    import shape
    outputpath = 'home/ubuntu/data/net1Predictions.txt'
    inputNamesPath = 'home/ubuntu/data/test_images_128_128_names.csv'
    names = read_csv(os.path.expanduser(inputNamesPath))  # load pandas dataframe
    #rename columns
    names.columns =list(range(0,len(names.columns)))
    with open(outputpath, "w") as outfile:
        outfile.write("image, level\n")    
    os.chdir(FTEST_PATH)
    j=0
    for fname in sorted(os.listdir('.'), key=os.path.getmtime):
        FTEST = FTEST_PATH+'/'+fname
        X, _ = load(test=True)
        y_pred = net1.predict(X)
        with open(outputpath, "a") as outfile:
            for i in range(1,shape(X)[0]):
                outfile.write( str(names[i+j]) +','+ (str(y_pred[i].astype('int').tolist()).strip('[]')))
                outfile.write("\n")
        j=j+shape(X)[0]
    os.chdir('../')




import cPickle as pickle
with open('home/ubuntu/data/net1.pickle', 'rb') as f:
    net1 = pickle.load( f )
y_pred = net1.predict(X)
    

import os
import shape
outputpath = 'home/ubuntu/data/net1Predictions.txt'
inputNamesPath = 'home/ubuntu/data/test_images_128_128_names.csv'
names = read_csv(os.path.expanduser(inputNamesPath))  # load pandas dataframe
#rename columns
names.columns =list(range(0,len(names.columns)))

with open(outputpath, "w") as outfile:
    outfile.write("image, level\n")
    for i in range(1,shape(X)[0]):
        outfile.write( str(names[i]) +','+ (str(y_pred[i].astype('int').tolist()).strip('[]')))
        outfile.write("\n")
#outfile.close