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

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = optparse.OptionParser('python trinity_pipeline.py [option]"')
    parser.add_option('--train',dest='train',help='train', default='')
    parser.add_option('--test',dest='test',help='test', default='')
    parser.add_option('--nu',dest='nu',help='nu', type="float", default=0.1)
    parser.add_option('--gamma',dest='gamma', type="float", help='gamma', default=0.1)
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

##########################################
## Master function
##########################################
def main():
    options = getOptions()
    train_x, train_y = extracData(options.train,'train')
    test_x, test_y = extracData(options.test,'test')  # test_y is all 0
    
    clf = OneClassSVM(nu=options.nu, kernel="rbf", gamma=options.gamma)
    clf.fit(train_x, train_y)
    train_pdt = clf.predict(train_x)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(train): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    test_pdt = clf.predict(test_x)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(test): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))  
    
    path, name = os.path.split(options.train)
    fileout = os.path.join(path,"outSub.csv")
    fout = open(fileout, 'w')
    fout.write("Id,WnvPresent\n")
    for index,eachy in enumerate(test_pdt):
        if int(eachy) == 1: 
            fout.write("%d,0\n" % (index+1))
        elif int(eachy) == -1:
             fout.write("%d,1\n" % (index+1))
    fout.close()
    
if __name__ == "__main__":
    main()