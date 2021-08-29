from os import path
import sys, json, time
import tensorflow as tf
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,minmax_scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense,Dropout
from keras.layers import TimeDistributed

# get data from json files
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# read input data
training_routes_pathIS = path.join(BASE_DIR, 'data/model_build_inputs/invalid_sequence_scores.json')
with open(training_routes_pathIS, newline='') as in_file11:
    invalid_score = json.load(in_file11)
training_routes_pathAS = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
with open(training_routes_pathAS, newline='') as in_file1:
    actual_sequence = json.load(in_file1)
training_routes_pathRD = path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
with open(training_routes_pathRD, newline='') as in_file2:
    route_data = json.load(in_file2)
training_routes_pathPD = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
with open(training_routes_pathPD, newline='') as in_file3:
    package_data = json.load(in_file3)

X_allR,route_stopsapp,route_highscore,Y_allR = [],[],[],[]


####_______________________________________________
## specify number of routes
route_data_sample = {k: route_data[k] for k, _ in zip(route_data,range(100))}

route_data_sample_HS = {h: route_data[h] for h, v in route_data.items() if v['route_score'] == 'High'}

route_data_subsample_HS = {k: route_data[k] for k, _ in zip(route_data_sample_HS,range(1000))}
# print(len(route_data_subsample_HS))

for krml,vrml in route_data.items():
    route_stopsml = route_data[krml]['stops']
    route_stopsapp.append(route_stopsml)
max_len = len(max(route_stopsapp, key=len))
print(max_len)

## start the train data with the depot



# print(route_data_sample)

for kr,vr in route_data.items():
    route_stops = route_data[kr]['stops']
    dic = {d for d, vd in route_stops.items() if vd['type'] == 'Station'}
    depot = list(dic)[0]
    index = list(route_stops.keys()).index(depot)
    stops_seq = actual_sequence[kr]['actual']
    # stops_seq_list = list(stops_seq)
    stops_seq_list = [(k,v) for k,v in stops_seq.items()]
    # print(stops_seq_list)
    # print(type(stops_seq_list))
    stops_seq_list.insert(0, stops_seq_list.pop(index))
    # print(stops_seq_list)
    target = [n[1] for n in stops_seq_list]

    stops_list = [{**value, **{'id':key}} for key, value in route_stops.items()]
    stops_list.insert(0, stops_list.pop(index))
    lbl_encode = LabelEncoder()
    stops_list_id = lbl_encode.fit_transform([i['id'] for i in stops_list])
    stops_list_zone = lbl_encode.fit_transform([i['zone_id'] for i in stops_list])
    stops_list_lat = [i['lat'] for i in stops_list]
    stops_list_lng = [i['lng']*-1 for i in stops_list]
    stops_list_features = list(zip(stops_list_id,stops_list_lat,stops_list_lng))
    stops_list_featuresnew = list(zip(stops_list_zone,stops_list_lat,stops_list_lng))
    # print(stops_list_featuresnew)


    df = DataFrame(stops_list_features)
    df2 = DataFrame(stops_list_featuresnew)
    y = df[2]
    X = df2
    x_norm = minmax_scale(df2.values)
    X1 = x_norm
    if len(X1)< max_len:
        added_pad = max_len - len(X1)
        # X1pad = np.pad(X1, [(0, added_pad), (0, added_pad)], mode='constant', constant_values= 5)
        X1pad = np.pad(X1,((0, added_pad), (0, 0)), mode='constant', constant_values= 0.1)
        targetpad = np.pad(target,(0, added_pad), mode='constant', constant_values= 0)
        X1 = X1pad
        target = targetpad

    X_allR.append(X1)
    Y_allR.append(target)

print(len(route_data))
X_allR= array(X_allR)
Y_allR= array(Y_allR).reshape(len(route_data),max_len,1)
Y_allRn = Y_allR[:,1:,:]

xTrain, xTest, yTrain, yTest = train_test_split(X_allR, Y_allR, test_size = 0.1, random_state = 0)

n_out = max_len

samples = len(route_data_sample_HS)
n_in = max_len
features = 3


# define model for several routes
model = Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.1,
 input_shape=(n_in, features)))
# model.add(LSTM(n_in, activation='tanh'))
model.add(Bidirectional(LSTM(64, activation='sigmoid',return_sequences=True)))
# model.add(LSTM(n_in, activation='tanh'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(n_out)))


model.add(Dense(1))
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
history = model.fit(xTrain, yTrain,validation_split=0.1, epochs=10, verbose=1)

model.summary()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate the model
scores = model.evaluate(xTest, yTest, verbose=0)

print(scores)
#
# demonstrate prediction
yhat = model.predict(xTest, verbose=1)

print(yhat)
print(yhat[0, :, :])


## final output structure
# serialize model to JSON
model_json = model.to_json()

# # Write output data
model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
with open(model_path, 'w') as out_file:
    json.dump(model_json, out_file)
    print("Success: The '{}' file has been saved".format(model_path))

model_path2=path.join(BASE_DIR, 'data/model_build_outputs/model.h5')
with open(model_path2, 'w') as out_file2:
    json.dump(model.save_weights(model_path2), out_file2)
    print("Success: The '{}' file has been saved".format(model_path2))


