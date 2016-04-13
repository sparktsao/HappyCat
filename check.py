import numpy
import scipy
import pandas
import sklearn
import theano
import keras
import h5py
info = []
import os
import platform
info.append("versions:")
info.append("os"+os.name+",")
info.append(platform.system())
info.append(platform.release())
info.append( "numpy"+numpy.__version__ )
info.append( "scipy"+scipy.__version__)
info.append( "pandas"+pandas.__version__)
info.append( "sklearn"+sklearn.__version__)
info.append( "theano"+theano.__version__)
info.append( "keras"+keras.__version__)
info.append( "h5py"+h5py.__version__)
print ",".join(info)





from keras.models import Sequential
from keras.layers.core import Dense
X_train = numpy.asarray([[0.1,0.2,0.3],[0.4,0.5,0.6]])
y_train = numpy.asarray([0,1])
model = Sequential()
model.add(Dense(output_dim=64,input_dim=X_train.shape[1], init='uniform'))
model.add(Dense(1))
from keras.optimizers import SGD
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error",optimizer=sgd)
model.fit(X_train,y_train)
json_string = model.to_json()
model.save_weights("check.hdf5",overwrite=True)

from keras.models import model_from_json
model2 = model_from_json(json_string)
model2.load_weights("check.hdf5")
print model2.predict(numpy.asarray([[0.1,0.2,0.3]]))
print "check success"
