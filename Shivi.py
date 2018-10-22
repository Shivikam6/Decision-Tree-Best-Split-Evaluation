import numpy as np
import math

D = np.loadtxt('train.txt', delimiter=',')
Data2 = np.loadtxt('test.txt', delimiter=',')

def IG(D1, index, value):   # to evaluate Information Gain
    class_Y = D1[1]
    list_Y = list(class_Y)
    C0 = list_Y.count(0)
    C1 = list_Y.count(1)
    attribute = D1[0]
    a_l = attribute[:, index]
    data_Y = list()
    data_N = list()

    n = len(list_Y)

    class_entropy = -((C0 / len(list_Y)) * np.log2(C0 / len(list_Y))) - ((C1 / len(list_Y)) * np.log2(C1 / len(list_Y)))    #Class Entropy
   # print(class_entropy)
    for i in range(len(a_l)): #Adding Class attributes 0 and 1 in list_Y and list_N
        if (a_l[i] <= value):
            data_Y.append(list_Y[i])
        else:
            data_N.append(list_Y[i])
    total_Y = len(data_Y)
    total_N = len(data_N)
    Y1_value = data_Y.count(1)
    N1_value = data_Y.count(0)
    Y2_value = data_N.count(1)
    N2_value = data_N.count(0)
    if (Y1_value != 0 and N1_value != 0):       #To check for Division by zero error
        entropyof1 = -((Y1_value / total_Y) * np.log2(Y1_value / total_Y)) - ((N1_value / total_Y) * np.log2(N1_value / total_Y))
    else:
        entropyof1 = 0
    if (Y2_value != 0 and N2_value != 0):       #To check for Division by zero error
        entropyof2 = -((Y2_value / total_N) * np.log2(Y2_value / total_N)) - ((N2_value / total_N) * np.log2(N2_value / total_N))
    else:
        entropyof2 = 0
    attribute_entropy = ((total_Y / n) * entropyof1) + ((total_N / n) * entropyof2)
    info_gain = class_entropy - attribute_entropy  # Calculating Information Gain

    return info_gain

def CART(D1, index, value): #implementing CART function
    data_X = D1[0]
    class_Y = D1[1]
    numEntries = len(data_X)
    y_values = 0  # no. of yes for whole data set

    for i in range(len(data_X)):
        if (class_Y[i]):
            y_values = y_values + 1

    n_Values = numEntries - y_values  # no. of no for whole data set
    column = data_X[:, index]
    # print(column)
    column_list = list(column)
    y_1 = 0  # number of YES for set of data which are less than or equal to value
    n_1 = 0  # number of NO for set of data which are less than or equal to value
    for i in range(len(data_X)):
        if (column_list[i] <= value):
            if (class_Y[i]):
                y_1 = y_1 + 1
            else:
                n_1 = n_1 + 1

    y_2 = y_values - y_1  # number of YES for set of data which are greater than value
    n_2 = n_Values - n_1  # number of NO for set of data which are greater than value
    print(y_1, y_2, n_1, n_2)
    if ((n_1) != 0 and (y_1) != 0):     #To check for Division by zero error
        left_Y = ((y_1) / (y_1 + n_1))
        left_N = ((n_1) / (y_1 + n_1))

    else:
        left_N = 0
        left_Y = 0
    if (n_2 != 0 and y_2 != 0):         #To check for Division by zero error
        right_Y = ((y_2) / (y_2 + n_2))
        right_N = ((n_2) / (y_2 + n_2))

    else:
        right_N = 0
        right_Y = 0

    P1 = left_Y - right_Y
    P2 = left_N - right_N

    Cart = 2 * ((y_1 + n_1) / numEntries) * ((y_2 + n_2) / numEntries) * (np.abs(P1) + np.abs(P2)) #Calculating CART by formula

    return Cart


def G(D1, index, value):    # Implementing Gini Index

    data_X = D1[0]
    class_Y = D1[1]
    numEntries = len(data_X)
    y_values = 0  # no. of yes for whole data set

    for i in range(len(data_X)):
        if (class_Y[i]):
            y_values = y_values + 1

    n_Values = numEntries - y_values  # number of NO for whole data set
    column = data_X[:, index]
    #print(column)
    column_list = list(column)
    y_1 = 0  # number of YES for set of data which are less than or equal to value
    n_1 = 0  # number of NO for set of data which are less than or equal to value
    for i in range(len(data_X)):
        if (column_list[i] <= value):
            if (class_Y[i]):
                y_1 = y_1 + 1
            else:
                n_1 = n_1 + 1

    y_2 = y_values - y_1  # number of YES for set of data which are greater than value
    n_2 = n_Values - n_1  # number of NO for set of data which are greater than value
    print(y_1,y_2,n_1,n_2)
    if ((n_1) != 0 and (y_1) != 0):         #To check for Division by zero error
        left_Y = ((y_1) / (y_1 + n_1))
        left_N = ((n_1) / (y_1 + n_1))
        temp1 = 1 - (np.square(left_Y) + np.square(left_N))
    else:
        temp1 = 0
    if (n_2 != 0 and y_2 != 0):             #To check for Division by zero error
        right_Y = ((y_2) / (y_2 + n_2))
        right_N = ((n_2) / (y_2 + n_2))
        temp2 = 1 - (np.square(right_Y) + np.square(right_N))
    else:
        temp2 = 0

    Gini = (((y_1 + n_1) / numEntries) * (temp1)) + (((y_2 + n_2) / numEntries) * (temp2))
    return Gini


def bestSplit(D1, criterion):
    li = D1[0]
    #print("li is",li)
    N = li.shape[1]
    #print("N is",N)
    c=set(li[:,0])
    #print("C is ",c)
    split_point = [-1, -1]
    if (criterion == 'IG'):     #Information Gain criterion
        ig = -1
        for i in range(0, N-1):
            c = set(li[:, i])
            for j in c:
                temp_ig = IG(D1, i, j)
                if (temp_ig > ig):      # better split points, higher Information Gain
                    ig = temp_ig
                    split_point = [i, j]
        return split_point

    if (criterion == 'G'):      # Gini Index criterion
        g = 2
        for i in range(0, N-1):
            c = set(li[:, i])
            #print(c)
            for j in c:
                temp_g = G(D1, i, j)
                if (temp_g < g):    #better split points, lower Gini Index
                    g = temp_g
                    split_point = [i, j]
        return split_point

    if (criterion == 'CART'):       # CART criterion
        cart = -1
        for i in range(0,N-1):
            c = set(li[:, i])
            #print("c is",c)
            for j in c:
                temp_c = CART(D1, i, j)
                if (temp_c > cart):     # better split points, higher CART
                    cart = temp_c
                    split_point = [i, j]
        return split_point


def load(filename):
    D = np.genfromtxt('train.txt', delimiter=',')

    c = len(D[0])
    #print("c is",c)
    y = D[:, c - 1]
    x = D[:, 0:c - 1]

    #print("x is\n",x)
    #print("y is\n",y)
    new_data = [x, y]
    return new_data

def classifyIG(train, test):
    #print(train)
    att_1 = bestSplit(train, 'IG')

    #a=list(att_1)
    attribute = int(att_1[0])
    value = int(att_1[1])
    #value = val
#    split_at = train[:,attribute]
    print("IG attribute in classify IG", attribute, "IG value in classify IG", value)
    #for value in attribute.keys():
    val_freq_1 = {}
    val_freq_2 = {}
    train = train[0]
    #split = a[attribute]
    record = []
    for record in train:
        print(record[:attribute])
        l=len(record[:attribute])
        print(l)
        for i in range(0,l-1):
            if(record[l] < value):
                if (record[l] in val_freq_1):
                    val_freq_1[record[l]] = val_freq_1[record[l]] + 1.0
                else:
                    val_freq_1[record[l]] = 1.0
            elif(record[l] > value):
                if (record[l] in val_freq_2):
                    val_freq_2[record[l]] = val_freq_2[record[l]] + 1.0
                else:
                    val_freq_2[record[l]] = 1.0
    data_subset1 = []
    data_subset2 = []

    for val in val_freq_1.keys():
        data_subset = [record for record in train if record[l] == val]
        data_subset1 = data_subset1 + data_subset
    for val in val_freq_2.keys():
        data_subset = [record for record in train if record[l] == val]
        data_subset2 = data_subset2 + data_subset

    frequency_1 = {}
    frequency_2 = {}

    for row in data_subset1:
        if (row[attribute] in frequency_1):
            frequency_1[row[attribute]] += 1.0
        else:
            frequency_1[row[attribute]] = 1.0

    for row in data_subset2:
        if (row[attribute] in frequency_2):
            frequency_2[row[attribute]] += 1.0
        else:
            frequency_2[row[attribute]] = 1.0

    listoftest = []
    for a in test:
        listoftest.append(a[attribute])

    predictedclass = []

    if (0 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(1)
            if (a > value):
                predictedclass.append(0)

    if (1 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(0)
            if (a > value):
                predictedclass.append(1)

    if (0 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[0] > frequency_1[1]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(0)
                if (a > value):
                    predictedclass.append(1)

    if (1 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[1] > frequency_1[0]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(1)
                if (a > value):
                    predictedclass.append(0)

    return (predictedclass)

'''def classifyG(train, test):
    a = bestSplit(train, 'G')
    attribute = a[0]
    val = a[1]
    value = val
    #    split_at = train[:,attribute]
    print("IG attribute in classify Gini", attribute, "IG value in classify Gini", value)
    # for value in attribute.keys():
    val_freq_1 = {}
    val_freq_2 = {}
    record = []
    for record in train:
        if (record[:attribute[i] < value):
            if (record[:attribute[i]] in val_freq_1):
                val_freq_1[record[:attribute[i]] = val_freq_1[record[:attribute[i]] + 1.0
            else:
                val_freq_1[record[:attribute[i]] = 1.0
        elif (record[:attribute[i]]> value):
            if (record[:attribute[i] in val_freq_2):
                val_freq_2[record[:attribute[i]] = val_freq_2[record[:attribute[i]] + 1.0
            else:
                val_freq_2[record[:attribute[i]] = 1.0
    data_subset1 = []
    data_subset2 = []

    for val in val_freq_1.keys():
        data_subset = [record for record in train if record[:attribute[i] == val]
        data_subset1 = data_subset1 + data_subset
    for val in val_freq_2.keys():
        data_subset = [record for record in train if record[:attribute[i] == val]
        data_subset2 = data_subset2 + data_subset

    frequency_1 = {}
    frequency_2 = {}

    for row in data_subset1:
        if (row[10] in frequency_1):
            frequency_1[row[10]] += 1.0
        else:
            frequency_1[row[10]] = 1.0

    for row in data_subset2:
        if (row[10] in frequency_2):
            frequency_2[row[10]] += 1.0
        else:
            frequency_2[row[10]] = 1.0

    listoftest = []
    for a in test:
        listoftest.append(a[attribute])

    predictedclass = []

    if (0 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(1)
            if (a > value):
                predictedclass.append(0)

    if (1 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(0)
            if (a > value):
                predictedclass.append(1)

    if (0 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[0] > frequency_1[1]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(0)
                if (a > value):
                    predictedclass.append(1)

    if (1 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[1] > frequency_1[0]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(1)
                if (a > value):
                    predictedclass.append(0)

    return (predictedclass)


def classifyCART(train, test):
    a = bestSplit(train, 'CART')
    attribute = a[0]
    val = a[1]
    value = val
    #    split_at = train[:,attribute]
    print("IG attribute in classify CART", attribute, "IG value in classify CART", value)
    # for value in attribute.keys():
    val_freq_1 = {}
    val_freq_2 = {}
    record = []
    for record[:attribute[i]] in train:
        if (record[:attribute[i]] <= value):
            if (record[:attribute[i]] in val_freq_1):
                val_freq_1[record[:attribute[i]]] = val_freq_1[record[:attribute[i]]] + 1.0
            else:
                val_freq_1[record[:attribute[i]]] = 1.0
        elif (record[:attribute[i]] > value):
            if (record[:attribute[i]] in val_freq_2):
                val_freq_2[record[:attribute[i]]] = val_freq_2[record[:attribute[i]]] + 1.0
            else:
                val_freq_2[record[:attribute[i]]] = 1.0
    data_subset1 = []
    data_subset2 = []

    for val in val_freq_1.keys():
        data_subset = [record for record in train if record[:attribute[i]] == val]
        data_subset1 = data_subset1 + data_subset
    for val in val_freq_2.keys():
        data_subset = [record for record in train if record[:attribute[i]] == val]
        data_subset2 = data_subset2 + data_subset

    frequency_1 = {}
    frequency_2 = {}

    for row in data_subset1:
        if (row[10] in frequency_1):
            frequency_1[row[10]] += 1.0
        else:
            frequency_1[row[10]] = 1.0

    for row in data_subset2:
        if (row[10] in frequency_2):
            frequency_2[row[10]] += 1.0
        else:
            frequency_2[row[10]] = 1.0

    listoftest = []
    for a in test:
        listoftest.append(a[attribute])

    predictedclass = []

    if (0 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(1)
            if (a > value):
                predictedclass.append(0)

    if (1 not in frequency_1.keys()):
        for a in listoftest:
            if (a <= value):
                predictedclass.append(0)
            if (a > value):
                predictedclass.append(1)

    if (0 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[0] > frequency_1[1]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(0)
                if (a > value):
                    predictedclass.append(1)

    if (1 in frequency_1.keys() and 1 in frequency_1.keys()):
        if (frequency_1[1] > frequency_1[0]):
            for a in listoftest:
                if (a <= value):
                    predictedclass.append(1)
                if (a > value):
                    predictedclass.append(0)

    return (predictedclass)'''


def main():
    filename = "train.txt"
    D1 = load(D)
    #print(D1)
    #D2 = load(Data2)
    #print(D1)
    '''col1 = input("Enter index for IG:\n")
    index1 = int(col1)
    val1 = input("Enter value for IG:\n")
    value1 = int(val1)
    ig = IG(D1, index1, value1)
    print("info gain:", ig)

    col2 = input("Enter column for Gini index:\n")
    index2 = int(col2)
    val2 = input("Enter value for Gini index:\n")
    value2 = int(val2)
    gini = G(D1, index2, value2)
    print("Gini Index:", gini)

    col3 = input("Enter column for CART:\n")
    index3 = int(col3)
    val3 = input("Enter value for CART:\n")
    value3 = int(val3)
    cart = CART(D1, index3, value3)
    print("CART is:", cart)

    tuple1 = bestSplit(D1, 'IG')
    print("Best split for IG", tuple1)
    tuple2 = bestSplit(D1, 'G')
    print("Best split for Gini", tuple2)
    tuple3 = bestSplit(D1, 'CART')
    print("Best split for CART", tuple3)

#    Data2 = np.genfromtxt('text.txt', delimiter=',')
    c1 = len(Data2[0])
    # print("c is",c)
    #y = Data2[:, c1]
    #x = Data2[:, 0:c1]

    # print("x is\n",x)
    # print("y is\n",y)
#    D2 = [x, y]
 #   print("D2 is",D2)'''
    classify_ig = classifyIG(D1, Data2)
    print("Predicted Class for Info Gain: ", classify_ig)

    classify_g = classifyG(D1, Data2)
    print("Predicted Class for Gini Index: ", classify_g)

    classify_cart = classifyCART(D1, Data2)
    print("Predicted Class for CART: ", classify_cart)

if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
main()
