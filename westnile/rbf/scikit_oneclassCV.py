#!/usr/bin/env python

import sys
import subprocess
import optparse
import shutil
import os
import csv
import sklearn
from sklearn.ensemble import *
from sklearn.pipeline import Pipeline
from sklearn.svm import *
import numpy as np
from sklearn.grid_search import *
import math
import copy

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = optparse.OptionParser('python trinity_pipeline.py [option]"')
    parser.add_option('--train',dest='train',help='train', default='')
    parser.add_option('--test',dest='test',help='test', default='')

    options, args = parser.parse_args()

    #print same
    if (options.train=='' or options.test==''):
        parser.print_help()
        print ''
        print 'You forgot to provide some data files!'
        print 'Current options are:'
        print options
        sys.exit(1)
    return options

def extracData(filename,filetype):
    fcsv = csv.reader(file(filename))
    x = []
    y = []
    for eachline in fcsv:
        if fcsv.line_num == 1:  
            continue
        if filetype == 'train':
            x.append(eachline[:-1])
            y.append(eachline[-1])
        elif filetype == 'test':
            x.append(eachline)
            y.append(-1)
    return x,y

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

def generateDatafile(filename, indices):
    
    path, name = os.path.split(filename)
    outname = name.split('.')[0] + "CV." + name.split('.')[1]
    
    fileout = os.path.join(path,outname)
    fout = open(fileout, 'w')

    fcsv = csv.reader(file(filename))
    for eachline in fcsv:
        linestr = ''
        for i in indices:
            linestr = linestr + eachline[i] + ','
        fout.write(linestr[:-1]+'\n')
    fout.close()

def trimfrq(des):
    desmap = map(list, zip(*des))
    desindex = []
    for index,eachdes in enumerate(desmap):
        eachdes = list(set(eachdes))
        if len(eachdes) == 1:
            desindex.append(index)
    return desindex
    
##########################################
## Master function
##########################################
def main():
    options = getOptions()
    
    train_x, train_y = extracData(options.train,'train')
    test_x, test_y = extracData(options.test,'test')  # test_y is all 0
    
    ftsel = ExtraTreesClassifier()
    ftsel.fit(train_x, train_y)
    
    frqIndex = trimfrq(train_x)
    
    train_x_new = ftsel.transform(train_x)
    test_x_new = ftsel.transform(test_x)
    importances = ftsel.feature_importances_
    indices_test = np.argsort(importances)[::-1]
    indices_test = indices_test.tolist()
    for i in frqIndex:
        indices_test.remove(i)

    
    indices_train = copy.deepcopy(indices_test)
    indices_train.append(-1)
    
    generateDatafile(options.train, indices_train)
    generateDatafile(options.test, indices_test)

    
    MCCmax = 0
    best_g = 0
    best_n = 0
    g_range = np.logspace(-100, 10, 111)
    n_range = np.linspace(0,1,21)[1:]
    for g in g_range:
        for n in n_range:
            clf = OneClassSVM(nu=n, kernel="rbf", gamma= g)
            clf.fit(train_x_new, train_y)
            train_pdt = clf.predict(train_x_new)
            MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
            if MCC > MCCmax:
                best_g = g
                best_n = n
                MCCmax = MCC
                print "MCCmax, best_n, best_g:"
                print "%s,%s,%s" % (str(MCCmax), str(best_n) , str(best_g))
                print "MCC, Acc_p , Acc_n, Acc_all(train): "
                print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
                    
                test_pdt = clf.predict(test_x_new)
                MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
                print test_pdt[-1]
                print "MCC, Acc_p , Acc_n, Acc_all(test): "
                print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))  
                print  

#     C_range = np.logspace(-10,  10, 21, base=2)
#     gamma_range = np.logspace(-10,  10, 21, base=2)         
#     param_grid = dict(gamma=gamma_range, C=C_range)
#       
#     grid = GridSearchCV(SVC(), param_grid=param_grid)
#     grid.fit(train_x, train_y)
#     print grid.best_params_
#         
#     clf = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
#     clf.fit(train_x, train_y)
#     train_pdt = clf.predict(train_x)
#     MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
#       
#     print "MCC, Acc_p , Acc_n, Acc_al(train)l: \n"
#     print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
#       
#     test_pdt = clf.predict(test_x)
#     MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
#       
#     print "MCC, Acc_p , Acc_n, Acc_all(test): \n"
#     print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    
#     clf = Pipeline([
#                     ("feature_selection", ExtraTreesClassifier()),
#                     ("classification", OneClassSVM())
#                    ]
#                   )
#     clf.fit(train_x, train_y)
      
if __name__ == "__main__":
    main()