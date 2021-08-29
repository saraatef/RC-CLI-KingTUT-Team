
from keras.models import model_from_json
from os import path
import sys, json, time
# lstm autoencoder predict sequence
from pandas import DataFrame, Series
import numpy as np
from numpy import array
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,minmax_scale


# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Input Data')
# Model Build output
model_path=path.join(BASE_DIR, 'data/model_build_outputs/model3.json')
with open(model_path, newline='') as in_file:
    model_build_out = json.load(in_file)
# Prediction Routes (Model Apply input)
prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
with open(prediction_routes_path, newline='') as in_file:
    prediction_routes = json.load(in_file)


X_allR,route_stopsapp,route_highscore,Y_allR,predseq,predseq2,outputallR = [],[],[],[],[],[],[]


####_______________________________________________
## specify number of routes
route_data_sample = {k: prediction_routes[k] for k, _ in zip(prediction_routes,range(100))}
route_data_subsample_HS = {k: prediction_routes[k] for k, _ in zip(route_data_sample,range(100))}
print(len(route_data_subsample_HS))

for krml,vrml in prediction_routes.items():
    route_stopsml = prediction_routes[krml]['stops']
    route_stopsapp.append(route_stopsml)
    lenm = len(route_stopsml)
    print(lenm)
max_len  = 222
print(max_len)

for kr,vr in prediction_routes.items():
    route_stops = prediction_routes[kr]['stops']
    dic = {d for d, vd in route_stops.items() if vd['type'] == 'Station'}
    depot = list(dic)[0]
    index = list(route_stops.keys()).index(depot)

    stops_list = [{**value, **{'id':key}} for key, value in route_stops.items()]
    stops_list.insert(0, stops_list.pop(index))
    lbl_encode = LabelEncoder()
    stops_list_id = lbl_encode.fit_transform([i['id'] for i in stops_list])
    stops_list_zone = lbl_encode.fit_transform([i['zone_id'] for i in stops_list])
    stops_list_lat = [i['lat'] for i in stops_list]
    stops_list_lng = [i['lng']*-1 for i in stops_list]
    stops_list_features = list(zip(stops_list_id,stops_list_lat,stops_list_lng))
    stops_list_featuresnew = list(zip(stops_list_zone,stops_list_lat,stops_list_lng))

    df = DataFrame(stops_list_features)
    df2 = DataFrame(stops_list_featuresnew)

    X = df2
    x_norm = minmax_scale(df2.values)
    X1 = x_norm
    if len(X1)< max_len:
        added_pad = max_len - len(X1)
        # X1pad = np.pad(X1, [(0, added_pad), (0, added_pad)], mode='constant', constant_values= 5)
        X1pad = np.pad(X1,((0, added_pad), (0, 0)), mode='constant', constant_values= 0.1)
        X1 = X1pad

    X_allR.append(X1)


X_allR= array(X_allR)

print(X_allR.shape)
##__________________________________________________________________________________________


# load json and create model

model_pathm = path.join(BASE_DIR, 'data/model_build_outputs/model2.json')
with open(model_pathm, newline='') as in_file:
    json_file = json.load(in_file)

loaded_model = model_from_json(json_file)
ypred = loaded_model.predict(X_allR, verbose=0)

outputr ={}
n=0
for kr,vr in prediction_routes.items():

    route_stops = prediction_routes[kr]['stops']
    stops = len(route_stops)
    # print(stops)
    dic = {d for d, vd in route_stops.items() if vd['type'] == 'Station'}
    depot = list(dic)[0]
    print(depot)
    index = list(route_stops.keys()).index(depot)
    stops_listpred = [{**value, **{'id':key}} for key, value in route_stops.items()]
    stops_listpred.insert(0, stops_listpred.pop(index))

    predstopsid = [i['id'] for i in stops_listpred]
    predicted_seq = abs(ypred[n, :stops, :]).reshape(stops)
    n=n+1

    predicted_seq = predicted_seq.tolist()
    y=0
    for i in stops_listpred:

        preSeq = [i['id'], predicted_seq[y]]
        y=y+1
        # print(preSeq)
        predseq.append(preSeq)
    print(predseq)
    ordered_predseq= predseq[:1] + sorted(predseq[1:], key=lambda x: x[1])
    print(ordered_predseq)
    u=0
    for ii in ordered_predseq:
        preSeq2 = [ii[0], u]
        u=u+1
        predseq2.append(preSeq2)


    output = {kr:{'proposed':{iii[0]: iii[1] for iii in predseq2}}}
    outputr.update(output)

    outputallR.append(output)

    predseq = []
    predseq2 = []
# out = {outputallR}
print('finalOutput',outputr)
print(type(outputr))

print('\nApplying answer with real model...')

print('Data sorted!')

# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
    json.dump(outputr, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')
