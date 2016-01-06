import pandas as pd
import os
import numpy

os.system("ls ./learning")

train = pd.read_csv("./learning/train.csv", header=0, parse_dates=[0])
# test = pd.read_csv("./learning/test.csv", header=0, parse_dates=[1])
data = train
data["Year"] = data.Date.map(lambda d: d.year)
data['Month'] = data.Date.map(lambda d: d.month)


print(data.pivot_table(rows = ['Year', 'Month'],values=['NumMosquitos',"WnvPresent"], aggfunc=[numpy.mean, len, numpy.sum]))