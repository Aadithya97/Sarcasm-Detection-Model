#import the appropriate libraries
import math
#numpy is used for mathematical functions
import numpy as np
#csv is used for csv file processing
import csv
from keras.models import Sequential
from keras.layers import Dense
import random
import math
from sklearn.metrics import precision_recall_curve

#create the input based on the final dataset
def read_input():
    ListA = [];
    Listb = [];
    ListC = [];
    
#non-sarcasm based input
    with open('../../data/nonsarcasm_final_features.csv', 'rb') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        init = 1
        for row in file:
            if(init == 1):
                init = 0
                continue
            temp = [float(i) for i in row]
            ListA.append(temp[0:10])
            Listb.append(0)

        #sarcasm based input
    with open('../../data/sarcasm_final_features.csv', 'rb') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        init = 1
        for row in file:
            if(init == 1):
                init = 0
                continue
            temp = [float(i) for i in row]
            ListA.append(temp[0:10])
            Listb.append(1)
    ListA = np.asarray(ListA)
    Listb = np.asarray(Listb)

    ListD = np.c_[ListA, Listb]
    np.random.shuffle(ListD)

    ListX = ListD[:,0:10]
    Listy = ListD[:,10]
    return ListX,Listy

[ListA,Listb] = read_input()
#At this stage, the input is succesfully read

num_rows, num_cols = ListA.shape
n_train = int(num_rows*0.80)
#use the 80:20 rule for dividing the dataset into training data and test data
ListX = ListA[0:n_train,:]
ListY = Listb[0:n_train]

X_valid = ListA[n_train:num_rows,:]
Y_valid = Listb[n_train:num_rows]
#At this stage, the data is processed

#set the number of nodes for each layer
#input
num_input = 10  

#Hidden
num1=5

#Output
num2=1           

model = Sequential()
model.add(Dense(output_dim = num1, input_dim=num_input, init='uniform', activation='relu')) 
model.add(Dense(output_dim = num2, init='uniform', activation='sigmoid'))

#To Configure Learning
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# To train for fixed epochs
model.fit(ListX, ListY, nb_epoch=10, batch_size=10)

#To evaluate accuracy
scores = model.evaluate(X_valid, Y_valid)

#To generate output predictions batch by batch
test_classes = model.predict_classes(X_valid)

p_label = test_classes
p_actual = Y_valid
tp,fp,tn,fn=0.0,0.0,0.0,0.0
for predicted,actual in zip(p_label,p_actual):
    if(predicted==actual and predicted==1): tp+=1
    elif(predicted==actual and predicted==0):tn+=1
    elif(predicted==0):fn+=1
    else:fp+=1

precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
fscore = 2*precision*recall/(precision+recall)

print "\n"
print "the precision is: ",precision
print "the recall is: ",recall
print "the f-measure is: ",fscore

print "the final accuracy is: ",scores[1]
