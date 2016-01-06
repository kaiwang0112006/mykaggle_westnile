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
from sklearn.tree import *
from sklearn.neighbors import *
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
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

def floatlist(eachline):
    tempx = []
    for vx in eachline:
        if not ',' in vx:
            vx_float = float(vx)
        else:
            vxlist = vx.split(',')
            vxstr = ''.join(vxlist) 
            vx_float = float(vxstr)
        tempx.append(vx_float)
    return tempx

def extracData(filename,filetype):
    fcsv = csv.reader(file(filename))
    x = []
    y = []
    for eachline in fcsv:
        if fcsv.line_num == 1:  
            continue
        if filetype == 'train':
            tempx = floatlist(eachline[:-1])
            x.append(tempx)
            y.append(int(eachline[-1])) 
            #x.append(eachline[:-1])
            #y.append(eachline[-1])
        elif filetype == 'test':
            tempx = floatlist(eachline)
            x.append(tempx)
            y.append(0)
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
    
    train_x_new = ftsel.transform(train_x)
    test_x_new = ftsel.transform(test_x)
    importances = ftsel.feature_importances_
    indices_test = np.argsort(importances)[::-1]
    indices_test = indices_test.tolist()
    
    frqIndex = trimfrq(train_x_new)
    
    for i in frqIndex:
        indices_test.remove(i)

    indices_train = copy.deepcopy(indices_test)
    indices_train.append(-1)
    
    outnameTrain = options.train.split('.')[0] + "CV." + options.train.split('.')[1]
    outnameTest = options.test.split('.')[0] + "CV." + options.test.split('.')[1]
    
    generateDatafile(options.train, indices_train)
    generateDatafile(options.test, indices_test)

    train_x_new , train_y= extracData(outnameTrain,'train')
    test_x_new , test_y= extracData(outnameTest,'test')

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(class_weight='auto'),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(class_weight='auto'),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(class_weight='auto'),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
    
    classtitle = ["KNeighborsClassifier",
                  "SVC",
                  "SVC weighted",
                  "SVC(gamma=2, C=1)",
                  "DecisionTreeClassifier",
                  "DecisionTreeClassifier weighted",
                  "RandomForestClassifier",
                  "RandomForestClassifier weighted",
                  "AdaBoostClassifier",
                  "GaussianNB",
                  "LDA",
                  "QDA"]
    
    for i in range(len(classtitle)):
        ctitle = classtitle[i]
        clf = classifiers[i]
        clf.fit(train_x_new, train_y)
        train_pdt = clf.predict(train_x_new)
        MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
        print ctitle+":"
        print "MCC, Acc_p , Acc_n, Acc_all(train): "
        print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
        test_pdt = clf.predict(test_x_new)
        MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
        print "MCC, Acc_p , Acc_n, Acc_all(test): "
        print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))        
        print
    #clf =  AdaBoostClassifier(base_estimator=SVC(),algorithm='SAMME')
    #clf = OneClassSVM(kernel="rbf")
#     clf = SVC(class_weight='auto')
#     clf.fit(train_x_new, train_y)
#     train_pdt = clf.predict(train_x_new)
#     MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
#     print "MCC, Acc_p , Acc_n, Acc_all(train): "
#     print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))

      
if __name__ == "__main__":
    main()