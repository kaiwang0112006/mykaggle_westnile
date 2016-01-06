from SimpleLasagneNN import *
import sys
import os
import subprocess
import optparse
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle
import copy
from lasagne import layers

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

def trimfrq(des):
    desmap = map(list, zip(*des))
    desindex = []
    for index,eachdes in enumerate(desmap):
        eachdes = list(set(eachdes))
        if len(eachdes) == 1:
            desindex.append(index)
    return desindex

def generateDatafile(filename, indices):
    
    path, name = os.path.split(filename)
    outname = filename
    
    fileout = os.path.join(path,outname)
    fout = open(outname, 'w')

    fcsv = csv.reader(file(filename))
    for eachline in fcsv:
        linestr = ''
        for i in indices:
            linestr = linestr + eachline[i] + ','

        fout.write(linestr[:-1]+'\n')
    fout.close()

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

def main():
    options = getOptions()
    train_x, train_y = extracData(options.train,'train')
    test_x, test_y = extracData(options.test,'test')  # test_y is all 0

    train_x_new = copy.deepcopy(train_x)
    test_x_new = copy.deepcopy(test_x)
    indices_test = range(len(train_x_new[0]))
    
    frqIndex = trimfrq(train_x)
    
    for i in frqIndex:
        indices_test.remove(i)

    indices_train = copy.deepcopy(indices_test)
    indices_train.append(-1)
    
    outnameTrain = "temptr.csv"
    outnameTest = "tempte.csv"
    
    generateDatafile(outnameTrain, indices_train)
    generateDatafile(outnameTest, indices_test)

    train_x_new , train_y= extracData(outnameTrain,'train')
    test_x_new , test_y= extracData(outnameTest,'test')
    
    X = np.array(train_x_new)
    y = np.array(train_y)
    input_size = len(train_x_new)
    learning_rate = theano.shared(np.float32(0.1))
    
    net = NeuralNet(
    layers=[  
        ('input', InputLayer),
         ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, input_size), 
    hidden1_num_units=256, 
    dropout1_p=0.4,
    hidden2_num_units=256, 
    dropout2_p=0.4,
    output_nonlinearity=sigmoid, 
    output_num_units=1, 

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=learning_rate,
    update_momentum=0.9,
    
    # Decay the learning rate
    on_epoch_finished=[
            AdjustVariable(learning_rate, target=0, half_life=4),
            ],

    # This is silly, but we don't want a stratified K-Fold here
    # To compensate we need to pass in the y_tensor_type and the loss.
    regression=True,
    y_tensor_type = T.imatrix,
    objective_loss_function = binary_crossentropy,
     
    max_epochs=32, 
    eval_size=0.1,
    verbose=1,
    )
    
    net.fit(X, y)
    

if __name__ == "__main__":
    main()