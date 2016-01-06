import pandas as pd
import os
import numpy
import csv
import matplotlib.pyplot as plt

def drawyear(datadict, year):
    data = datadict[year]
    
    monthlist = sorted(data.keys())
    wnvlist = []
    numlist = []
    for m in monthlist:
        wnvlist.append(data[m]["wnv"])
        numlist.append(data[m]["NumMosquitos"])
    
    fig, ax1 = plt.subplots()
    ax1.set_title(year)
    l1 = ax1.plot(monthlist, wnvlist, 'b-', label="wnv")
    ax1.set_xlabel('month')
    ax1.set_ylabel('wnv frequency', color='b', )
    
    ax2 = ax1.twinx()
    l2 = ax2.plot(monthlist, numlist, 'r-', label="NumMosquitos")
    ax2.set_ylabel('NumMosquitos', color='r')
    plt.legend((l1, l2))
    
    outname = year + ".png"
    plt.savefig(outname)
    
    
    
# train = pd.read_csv("train.csv", header=0, parse_dates=[0])
# # test = pd.read_csv("./learning/test.csv", header=0, parse_dates=[1])
# data = train
# data["Year"] = data.Date.map(lambda d: d.year)
# data['Month'] = data.Date.map(lambda d: d.month)
#  
#  
# print(data.pivot_table(index = ['Year', 'Month'],values=['NumMosquitos',"WnvPresent"], aggfunc=[numpy.mean, len, numpy.sum]))
yearDict = {}
month = []
wnv = []
NumMosquitos = []
 
with open('train.csv') as csvfile:
    fcsv = csv.reader(csvfile)
    for eachline in fcsv:
        if fcsv.line_num == 1:  
            continue 
        date = eachline[0].split('-')
        year = date[0]
        month = int(date[1])
        if not yearDict.has_key(year):
            yearDict[year] = {}
        if not yearDict[year].has_key(month):
            yearDict[year][month] = {'wnv':0, 'NumMosquitos':0}
        yearDict[year][month]['wnv'] += int(eachline[11])
        yearDict[year][month]['NumMosquitos'] += int(eachline[10]) 
        
for year in yearDict:
    drawyear(yearDict,year)


         
        