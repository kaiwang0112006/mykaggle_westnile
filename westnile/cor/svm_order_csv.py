import copy

def read_file (file):
    all=[]
    for eachline in file:
        line=eachline.strip()
        line=line.split()
        all.append(line[0])
    return all

print "The program will rank the descriptors by a given order      --        Author:  Wangkai"
print
filename1=raw_input("Input file name for descriptor data: ")
fobj=open(filename1,'r')
filename2=raw_input("Input file name for rank order: ")
fobj2=open(filename2,'r')
filename3=raw_input("Input file name for result: ")
fobj3=open(filename3,'w')
allInput=[(line.strip()).split(',') for line in fobj.readlines()]
title_Input=allInput[0]
DESCRIPTOR=len(allInput[0])
COMPOUND=len(allInput)
data=[allInput[i] for i in range(1,COMPOUND)]
rank_order=read_file(fobj2)
long=len(rank_order)
fobj.close()
fobj2.close()
order=[]

for i in range(long):
    for j in range(DESCRIPTOR-1):
        if title_Input[j]==rank_order[i]:
            order.append(j)
            break

order.append(DESCRIPTOR-1)
lenth=len(order)
for i in range(COMPOUND):
    for j in range(lenth):
        fobj3.write(allInput[i][order[j]])
        fobj3.write('%s' % ',' if j!=lenth-1 else '\n')

#print "%s is calculate %s \t" % str(i),str(j)
fobj3.close()
raw_input("Press enter to exit")



