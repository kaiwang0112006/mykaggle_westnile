import sys
import subprocess
import optparse
import shutil
import os
import csv
import math

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = optparse.OptionParser('python trinity_pipeline.py [option]"')
    parser.add_option('--in',dest='input',help='input', default='')
    parser.add_option('--out',dest='output',help='output', default='output')

    options, args = parser.parse_args()

    #print same
    if (options.input == ''):
        parser.print_help()
        print ''
        print 'You forgot to provide some data files!'
        print 'Current options are:'
        print options
        sys.exit(1)
    return options

def main():
    options = getOptions()
    fin = open(options.input)
    fout = open(options.output, 'w')
    
    for eachline in fin:
        line = eachline.strip()
        if line[-1] == '0':
            newline = line[:-1] + '1\n'
        elif line[-1] == '1':
            newline = line[:-1] + '-1\n'
        else:
            newline = eachline
        fout.write(newline)
    fin.close()
    fout.close()

if __name__ == "__main__":
    main()