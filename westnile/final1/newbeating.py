"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from geopy.distance import great_circle
import csv
import re
import datetime
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import *
import math
import copy

Codedict = {'+FC':1, 'FC':2, 'TS':3, 'GR':4, 'RA':5, 'DZ':6, 'SN':7, 'SG':8, 'GS':9, 'PL':10, 'IC':11, 'FG+':12, 'FG':13, 'BR':14, 'UP':15, 'HZ':16, 'FU':17, 'VA':18, 'DU':19, 'DS':20, 'PO':21, 'SA':22, 'SS':23, 'PY':24, 'SQ':25, 'DR':26, 'SH':27, 'FZ':28, 'MI':29, 'PR':30, 'BC':31, 'BL':32, 'VC':33}
loclist = [[41.995,-87.933], [41.786,-87.752]]
speciesdict = {'CULEX PIPIENS/RESTUANS':1,'CULEX RESTUANS':2,'CULEX PIPIENS':3, 'CULEX SALINARIUS':4, 'CULEX TERRITANS':5, 'CULEX TARSALIS':6, 'CULEX ERRATICUS':7, 'UNSPECIFIED CULEX':8}
#traintitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed',"Tmax-1","Tmin-1","Tavg-1","DewPoint-1","WetBulb-1","PrecipTotal-1","Depart-1","Tmax-2","Tmin-2","Tavg-2","DewPoint-2","WetBulb-2","PrecipTotal-2","Depart-2","Tmax-3","Tmin-3","Tavg-3","DewPoint-3","WetBulb-3","PrecipTotal-3","Depart-3","Tmax-5","Tmin-5","Tavg-5","DewPoint-5","WetBulb-5","PrecipTotal-5","Depart-5","Tmax-7","Tmin-7","Tavg-7","DewPoint-7","WetBulb-7","PrecipTotal-7","Depart-7","Tmax-8","Tmin-8","Tavg-8","DewPoint-8","WetBulb-8","PrecipTotal-8","Depart-8","Tmax-12","Tmin-12","Tavg-12","DewPoint-12","WetBulb-12","PrecipTotal-12","Depart-12","Tmax-14","Tmin-14","Tavg-14","DewPoint-14","WetBulb-14","PrecipTotal-14","Depart-14",'spray', 'WnvPresent']
#testtitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed',"Tmax-1","Tmin-1","Tavg-1","DewPoint-1","WetBulb-1","PrecipTotal-1","Depart-1","Tmax-2","Tmin-2","Tavg-2","DewPoint-2","WetBulb-2","PrecipTotal-2","Depart-2","Tmax-3","Tmin-3","Tavg-3","DewPoint-3","WetBulb-3","PrecipTotal-3","Depart-3","Tmax-5","Tmin-5","Tavg-5","DewPoint-5","WetBulb-5","PrecipTotal-5","Depart-5","Tmax-7","Tmin-7","Tavg-7","DewPoint-7","WetBulb-7","PrecipTotal-7","Depart-7","Tmax-8","Tmin-8","Tavg-8","DewPoint-8","WetBulb-8","PrecipTotal-8","Depart-8","Tmax-12","Tmin-12","Tavg-12","DewPoint-12","WetBulb-12","PrecipTotal-12","Depart-12","Tmax-14","Tmin-14","Tavg-14","DewPoint-14","WetBulb-14","PrecipTotal-14","Depart-14",'spray']
#traintitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','spray', 'WnvPresent']
#testtitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','spray']


def readweather(fin):
    weatherdict = {}

    for dline in fin:
        if fin.line_num == 1:  
            continue  
        loc = dline[0]
        date = dline[1]
        if not weatherdict.has_key(date):
            weatherdict[date] = []
        templist = [loclist[int(dline[0])-1][0],loclist[int(dline[0])-1][1]]

        for i in range(2,len(dline)):
            if i == 12:
                code = 0.0
                codestr = dline[i].split()
                codestr = ''.join(codestr)
                codelist = findlist(codestr)
                for c in codelist:
                    code += float(Codedict[c])
                templist.append(code)
            else:
                if dline[i] == 'M' or dline[i] == '-' or dline[i] == 'T' or dline[i] == ' T' or dline[i] == '  T':
                    templist.append(-1.0)
                else:
                    try:
                        float(dline[i])
                    except:
                        print dline
                    templist.append(float(dline[i]))

        weatherdict[date].append(templist)

    return weatherdict

def findlist(codestr):
    tempstr = ''
    templist = []
    for s in codestr:
        tempstr +=s
        if tempstr in Codedict.keys():
            templist.append(tempstr)
            tempstr = ''
    return templist

def nearloc(Lat1, Long1):
    disdict = {}
    for i in range(len(loclist)):
        Lat2 = loclist[i][0]
        Long2 = loclist[i][1]
        dis = great_circle((Lat1, Long1), (Lat2,Long2)).kilometers
        #dis = math.sqrt((float(Lat2) - float(Lat1))**2 + (float(Long2) - float(Long1))**2)
        disdict[dis] = i

    mindis = sorted(disdict.keys())[0]
    return disdict[mindis]

def writecsvtitle(file, tlist):
    for i in range(len(tlist)):
        file.write("%s%s" % (tlist[i], (',' if i!=len(tlist)-1 else '\n') ))

def normalize(x, mean=None, std=None):
    x_nor = copy.deepcopy(x)
    flist = [[] for i in x[0]]
    
    for eachline in x:
        for f in range(len(eachline)):
            flist[f].append(eachline[f])
    
    count = len(x)
    if mean is None:
        mean = []
        for i in flist:
            mean.append(np.mean(i))
    if std is None:
        std = []
        for i in flist:
            std.append(np.std(i))
    for i in range(count):
        for j in range(len(x[i])):
            x_nor[i][j] = (x[i][j]-mean[j])/std[j]
    return x_nor, mean, std

def readspray(fin):
    sdict = {}
    for sline in fin:
        if fin.line_num == 1:  
            continue  
        date = sline[0]
        slat = sline[2]
        slong = sline[3]
        if not sdict.has_key(date):
            sdict[date] = []
        sdict[date].append((slat, slong))
    return sdict

def nearspray(spraylist,Lat1,Long1):
    for cor in spraylist:
        Lat2 = cor[0]
        Long2 = cor[1]
        dis = great_circle((Lat1, Long1), (Lat2,Long2)).kilometers
        if dis < 10:
            return 1
    return 0

def indexTodata(data, indices):
    newdata = []
    for eachline in data:
        temp = []
        for i in indices:
            try:
                temp.append(eachline[i])
            except:
                print i 
                print eachline
                print len(eachline)
                exit(0)
        newdata.append(temp)
        
    return newdata

 
def trimfrq(des):
    desmap = map(list, zip(*des))
    desindex = []
    for index,eachdes in enumerate(desmap):
        eachdes = list(set(eachdes))
        if len(eachdes) == 1:
            desindex.append(index)
    return desindex

def get_Accs(ty,pv):
    if len(ty) != len(pv):
        raise ValueError("len(ty) must equal to len(pv)")
    tp = tn = fp = fn = 0
    for v, y in zip(pv, ty):
        if int(y) == int(v):
            if int(y) == 1:
                tp += 1
            else:
                tn += 1
        else:
            if int(y) == 1:
                fn +=1
            else:
                fp +=1
    tp=float(tp)
    tn=float(tn)
    fp=float(fp)
    fn=float(fn)

    MCC_x = tp*tn-fp*fn
    a = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    MCC_y = float(math.sqrt(a))
    if MCC_y == 0:
        MCC = 0
    else:
        MCC = float(MCC_x/MCC_y)
    try:
        Acc_p=tp/(tp+fn)
    except:
        Acc_p=0
        
    try:
        Acc_n=tn/(tn+fp)
    except:
        Acc_n=0
        
    Acc_all = (tp + tn)/(tp + tn + fp + fn)
    return (MCC, Acc_p , Acc_n, Acc_all)

def main():
    # Load dataset 
    ftrain = csv.reader(file(r'../input/train.csv'))
    ftest = csv.reader(file(r'../input/test.csv'))
    fweather = csv.reader(file(r'../input/weather.csv'))
    fspray = csv.reader(file(r'../input/spray.csv'))
    
    weatherPasstimelist = ["Tmax","Tmin","Tavg","DewPoint","WetBulb","PrecipTotal","Depart"]
    weatherPasstimevalue = [2,3,4,6,7,16,5]
    
    weatherdict = readweather(fweather)
    spraydict = readspray(fspray)
    
    #generate train and test data
    print "generate train and test data"
    trout = []
    train_y = []
    for trlist in ftrain:
        templine = []
        if ftrain.line_num == 1:  
            continue  
        date = trlist[0]
        datelist = date.split('-')
        dateformate = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        Latitude = trlist[7]
        Longitude = trlist[8]
        Species = speciesdict[trlist[2]]

        AddressAccuracy = trlist[9]
        NumMosquitos = trlist[10]
        WnvPresent = trlist[11]
        train_y.append(WnvPresent)
        #write weather
        locid = nearloc(Latitude, Longitude)
        weatherlist = weatherdict[date][locid]
        templine.append(float(Species))
        for w in weatherlist[2:]:
            templine.append(float(w))

        #time before 1,3,7,14 days
#         passstr = ''
#         for days_ago in [1,2,3,5,8,12]:
#             day = dateformate - datetime.timedelta(days=days_ago)
#             weatherlistPasstime = weatherdict[str(day)][locid]
#             for obs in weatherPasstimevalue:
#                 try:
#                     templine.append(float(weatherlistPasstime[obs]))
#                 except:
#                     print weatherlistPasstime
#                     exit(0)

        templine.append(float(Latitude))
        templine.append(float(Longitude))
        
        #write spray
        if not spraydict.has_key(date):
            sprayvalue = 0
        else:
            if nearspray(spraydict[date],Latitude,Longitude):
                sprayvalue = 1
            else:
                sprayvalue = 0
        templine.append(sprayvalue)
        trout.append(templine)
    
    teout = []
    for telist in ftest:
        templine = []
        if ftest.line_num == 1:  
            continue  
        date = telist[1]
        dateformate = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        datelist = date.split('-')
        Latitude = telist[8]
        Longitude = telist[9]
        Species = speciesdict[telist[3]]
        locid = nearloc(Latitude, Longitude)
        weatherlist = weatherdict[date][locid]
        templine.append(float(Species))
        for w in weatherlist[2:]:
            templine.append(float(w))
        
        #time before 1,3,7,14 days
#         passstr = ''
#         for days_ago in [1,2,3,5,8,12]:
#             day = dateformate - datetime.timedelta(days=days_ago)
#             weatherlistPasstime = weatherdict[str(day)][locid]
#             for obs in weatherPasstimevalue:
#                 templine.append(float(weatherlistPasstime[obs]))

        templine.append(float(Latitude))
        templine.append(float(Longitude))
        #write spray
        if not spraydict.has_key(date):
            sprayvalue = 0
        else:
            if nearspray(spraydict[date],Latitude,Longitude):
                sprayvalue = 1
            else:
                sprayvalue = 0
        templine.append(sprayvalue)
        teout.append(templine)
 
    #remove feature with no distinction
    indices = [i for i in range(len(trout[0]))]
    frqIndex = trimfrq(trout)

    for i in frqIndex:
        indices.remove(i)
    
    #modeling
    print "modeling"
    train_x = indexTodata(trout, indices)
    test_x = indexTodata(teout, indices)
    ntr = np.array(train_x)
    
    train_x_nor, mean, std = normalize(train_x)
    test_x_nor, mean, std = normalize(test_x, mean, std)
    
    clf = RandomForestClassifier(class_weight='auto')
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators":[10,40]}
     
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
    grid_search.fit(train_x, train_y)
    clfOpt = RandomForestClassifier(bootstrap=grid_search.best_params_['bootstrap'],
                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                    min_samples_split=grid_search.best_params_['min_samples_split'],
                                    criterion=grid_search.best_params_['criterion'],
                                    max_features=grid_search.best_params_['max_features'],
                                    max_depth=grid_search.best_params_['max_depth'],
                                    class_weight='auto')

    clfOpt.fit(train_x_nor, train_y)
    train_pdt = clfOpt.predict(train_x_nor)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
    print "randomtree:"
    print "MCC, Acc_p , Acc_n, Acc_all(train): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    
    #predict test data
    print "predict test data"
    test_pdt = clfOpt.predict(test_x_nor)
    fprt = open('sampleSubmissionbyKW.csv','w')
    fprt.write("ID,WnvPresent\n")
    id = 1
    for eachy in test_pdt:
        fprt.write("%s,%s\n" % (str(id), str(eachy)))
    fprt.close()
      
if __name__ == "__main__":
    main()