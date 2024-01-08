
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import seaborn as sns

"""
create NN
2 hidden layers 
"""
def createModel():
    
    model = Sequential()

    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


"""
fit the model to the data X and classes Y, for a given number of epoch
"""
def fitModel(model,X,Y,epoch):
    model.fit(X,Y,epoch=epoch,batch_size = 10)
