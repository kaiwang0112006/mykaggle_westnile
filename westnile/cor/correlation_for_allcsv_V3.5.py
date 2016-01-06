ROW=COLUMN=0

import copy

def read_file (file):
    global ROW,COLUMN
    all=[]
    for eachline in file:
        line=eachline.strip()
        line=line.split()
        COLUMN+=1
        all.append(line)
    ROW=len(all[0])
    return all

def order_by_abs (list):
    long=len(list)
    for i in range(long):
        for j in range(i+1,long):
            if abs(list[i])<abs(list[j]):
                list[i],list[j]=list[j],list[i]
    return list

def correlation (data,i,j):
    long=COMPOUND-1
    sum_x=0.0
    sum_y=0.0
    sum_xy=0.0
    x_sqrt=0.0
    y_sqrt=0.0
    for x in range(long):
        sum_x+=float(data[x][i])
        try:
            float(data[x][j])
        except:
            print (data[x][j])
            print x
            print j
            raw_input("exit")
            exit()
        sum_y+=float(data[x][j])
        sum_xy+=float(data[x][i])*float(data[x][j])
        x_sqrt+=(float(data[x][i]))**2
        y_sqrt+=(float(data[x][j]))**2
    
    result=(sum_xy-(sum_x*sum_y)/long)/((x_sqrt-(sum_x**2)/long)*(y_sqrt-(sum_y**2)/long))**0.5
    return result

def frequency (data):
    result=[]
    row=len(data[0])
    column=len(data)
    end=0
    for i in range(row):

        count=0
        content=data[0][i]
        for j in range(column):
            try :
                data[j][i]
            except:
                print j
                print i
            if data[j][i]==content:
                count+=1
          
        if count<column-1:
            result.append(i)
        else:
            fobj4.write(title_Input[i]+' , ')
            d_delete.append(i)
            d_title.append(title_Input[i])
            end+=1
    if end==0:
        fobj4.write('None')
    return result
            

print "DESCRIPTORS_FILTER V3.5                      Author: WangKai" 
print "Note:"
print '''      1. Path contain no Chinese.\n      2. Make sure all the columns have been delete except descriptor data and activity.\n      3.The last column should be ACTIVITY\n'''

filename1=raw_input('Input filename for input : ')
filename3='desriptors_for_delete.txt'
filename4=raw_input('Input a name for Output : ')
act_corr=float(raw_input('Input Number of activity correlation : '))
des_corr=float(raw_input('Input Number of descriptor correlation : '))
print '''Correlation result will be saved in file "correlation.txt" \n'''  
fobj4 = open(filename3, 'w')
filename2='correlation.txt'
fobj1=open(filename1,'r')
fobj2=open(filename2,'w')

allInput=[(line.strip()).split(',') for line in fobj1.readlines()]
title_Input=allInput[0]
COMPOUND=len(allInput)
DESCRIPTOR=len(allInput[0])

d_title=[]
d_delete=[]




data=[allInput[i] for i in range(1,COMPOUND)]
fobj4.write('Descriptors deleted by frequency : \n')
frequency_com=frequency(data)


#writing title
for i in range(DESCRIPTOR):
    if i in frequency_com:
        fobj2.write(title_Input[i])
        if i!=DESCRIPTOR-1:
            fobj2.write('\t')
        else:
            fobj2.write('\n')
        
#compute the correlation
for i in range(DESCRIPTOR):
    print "Computing correlation of descriptor %d" % i
    for j in range(DESCRIPTOR):
        if i in frequency_com and j in frequency_com:
            r=correlation(data,i,j)
            fobj2.write(str(r))
            if j!=DESCRIPTOR-1:
                fobj2.write('\t')
            else:
                fobj2.write('\n')
    

print "correlation have been computed successfully!"

fobj1.close()
fobj2.close()




print "This programm will delete all the descriptors that activity correlation less then %lf and descriptors correlation more than %lf\n" % (act_corr,des_corr)
print "Result will be saved in desriptors_for_delete.txt"
fobj3 = open(filename2, 'r')


all=[]
all=read_file(fobj3)
d_delete=[]
title=all[0]
activity=[float(all[COLUMN-1][i]) for i in range(COLUMN-2)]
descriptor=[all[i] for i in range(1,COLUMN-1)]


fobj4.write("\n\ndescriptors deleted by activity correlation:\n")
count=0


#delete descriptors by activity correlation
for i in range(COLUMN-2):
    if abs(activity[i])<=act_corr:
        d_delete.append(i)
#        fobj2.write(title[i]+'is deleted because its coreelation with activity is: '+str(activity[i])+'\n')
        fobj4.write(title[i]+', ')
        d_title.append(title[i])
        count+=1
if count==0:
    fobj4.write("None\n")
        

order=[i for i in range(COLUMN-2)]
act_inorder=copy.deepcopy(activity)
act_inorder=order_by_abs(act_inorder)

for i in range(COLUMN-2):
    for j in range(COLUMN-2):
        if activity[j]==act_inorder[i]:
            order[i]=j
            break

        
long1=len(d_delete)

for i in range(COLUMN-2):
    for j in range(COLUMN-2):
        if order[i] not in d_delete and j not in d_delete:
            if (float(descriptor[j][order[i]]))>=des_corr:
                if (activity[order[i]])>(activity[j]):
                    d_delete.append(j)
                    print title[order[i]]+' - '+title[j]+' : '+descriptor[j][order[i]]
                    print str(activity[order[i]])+'   '+str(activity[j])
                elif (activity[order[i]])< (activity[j]):
                    d_delete.append(order[i])
                    print title[order[i]]+' - '+title[j]+' : '+descriptor[j][order[i]]
                    print str(activity[order[i]])+'   '+str(activity[j])
                else:
                    print title[order[i]]+"is equal with"+title[i]

   
long2=len(d_delete)
fobj4.write('\n\ndescriptors deleted by descriptors correlation:\n')
count=0
for i in range(long2):
    if i>=long1:
        fobj4.write(title[d_delete[i]])
        fobj4.write('%s' % (' , ' if i<long2-1 else ' .'))
        d_title.append(title[d_delete[i]])
        count+=1
if count==0:
    fobj4.write("None\n")
    


fobj3.close()
fobj4.close()

d_delete=[]
x=len(d_title)
y=len(title_Input)

for i in range(x):
    for j in range(y):
        if d_title[i]==title_Input[j]:
            d_delete.append(j)
            break


fobj5=open(filename4,'w')

for i in range(COMPOUND):
    linestr = ''
    for j in range(DESCRIPTOR):
        if j not in d_delete:
            if i==0:
                linestr = linestr + title_Input[j]+','
                #fobj5.write(title_Input[j]+',')
            else:
                linestr = linestr + allInput[i][j]+','
                #fobj5.write(allInput[i][j]+',')
    if j==DESCRIPTOR-1:
        linestr = linestr[:-1] + '\n'
        fobj5.write(linestr)

   
fobj5.close()

raw_input('Press enter to exit')