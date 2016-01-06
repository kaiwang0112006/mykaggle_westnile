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

Codedict = {'+FC':1, 'FC':2, 'TS':3, 'GR':4, 'RA':5, 'DZ':6, 'SN':7, 'SG':8, 'GS':9, 'PL':10, 'IC':11, 'FG+':12, 'FG':13, 'BR':14, 'UP':15, 'HZ':16, 'FU':17, 'VA':18, 'DU':19, 'DS':20, 'PO':21, 'SA':22, 'SS':23, 'PY':24, 'SQ':25, 'DR':26, 'SH':27, 'FZ':28, 'MI':29, 'PR':30, 'BC':31, 'BL':32, 'VC':33}
loclist = [[41.995,-87.933], [41.786,-87.752]]
speciesdict = {'CULEX PIPIENS/RESTUANS':1,'CULEX RESTUANS':2,'CULEX PIPIENS':3, 'CULEX SALINARIUS':4, 'CULEX TERRITANS':5, 'CULEX TARSALIS':6, 'CULEX ERRATICUS':7, 'UNSPECIFIED CULEX':8}
traintitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','spray', 'WnvPresent']
testtitle = ['Species', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','spray']

def readweather(fin):
    weatherdict = {}

    for dline in fin:
        if fin.line_num == 1:  
            continue  
        #dline = eachline.strip().split(',')
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
        disdict[dis] = i

    mindis = sorted(disdict.keys())[0]
    return disdict[mindis]

def writecsvtitle(file, tlist):
    for i in range(len(tlist)):
        file.write("%s%s" % (tlist[i], (',' if i!=len(tlist)-1 else '\n') ))

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

def main():
    # Load dataset 
    ftrain = csv.reader(file(r'train.csv'))
    ftest = csv.reader(file(r'test.csv'))
    fweather = csv.reader(file(r'weather.csv'))
    fspray = csv.reader(file(r'spray.csv'))
    
    weatherdict = readweather(fweather)
    spraydict = readspray(fspray)
    
    trout = open('outtrain.csv','w')
    writecsvtitle(trout, traintitle)
    for trlist in ftrain:
        if ftrain.line_num == 1:  
            continue  
        #trlist = trline.strip().split(',')
        date = trlist[0]
        Latitude = trlist[7]
        Longitude = trlist[8]
        Species = speciesdict[trlist[2]]
        AddressAccuracy = trlist[9]
        NumMosquitos = trlist[10]
        WnvPresent = trlist[11]
        
        #write weather
        locid = nearloc(Latitude, Longitude)
        weatherlist = weatherdict[date][locid]
        #trout.write("%s,%s,%s,%s,%s,%s,%s" % (date, str(Latitude), str(Longitude), str(Species), str(AddressAccuracy), str(NumMosquitos), str(WnvPresent)))
        trout.write("%s" % (str(Species)))

        for w in weatherlist[2:]:
            trout.write(",%s" % str(w)) 
            
        #write spray
        if not spraydict.has_key(date):
            sprayvalue = 0
        else:
            if nearspray(spraydict[date],Latitude,Longitude):
                sprayvalue = 1
            else:
                sprayvalue = 0
                
        trout.write(',%s,%s\n' % (str(sprayvalue),str(WnvPresent)))
        
    trout.close()
    
    teout = open('outtest.csv','w')
    writecsvtitle(teout, testtitle)
    for telist in ftest:
        if ftest.line_num == 1:  
            continue  
        #telist = trline.strip().split(',')
        date = telist[1]
        Latitude = telist[8]
        Longitude = telist[9]
        Species = speciesdict[telist[3]]

        locid = nearloc(Latitude, Longitude)
        weatherlist = weatherdict[date][locid]
        #trout.write("%s,%s,%s,%s,%s,%s,%s" % (date, str(Latitude), str(Longitude), str(Species), str(AddressAccuracy), str(NumMosquitos), str(WnvPresent)))
        teout.write("%s" % (str(Species)))
        for w in weatherlist[2:]:
            teout.write(",%s" % str(w)) 
            
        #write spray
        if not spraydict.has_key(date):
            sprayvalue = 0
        else:
            if nearspray(spraydict[date],Latitude,Longitude):
                sprayvalue = 1
            else:
                sprayvalue = 0
                
        teout.write(',%s\n' % (str(sprayvalue)))

    teout.close()

    
main()