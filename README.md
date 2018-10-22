# Decision-Tree-Best-Split-Evaluation
Decision Tree - Best Split Evaluation using Information Gain, GINI Index and Cart

Decision trees As part of this question you will implement and compare the Information Gain, Gini Index and CART evaluation measures for splits in decision tree construction.Let D = (X, y); |D| = n be a dataset with n samples. The entropy of the dataset is defined as

                      H(D) = -(summation for i=1 to 2)P(ci|D)log2P(ci|D)
                      
where P(ci|D) is the fraction of samples in class i. A split on an attribute of the form Xi <= c partitions the
dataset into two subsets DY and DN based on whether samples satisfy the split predicate or not respectively.
The split Entropy is the weighted average Entropy of the resulting datasets DY and DX:

                      H(DY, DN) =(nY/n)H(DY) +(nN/n)H(DN)

where nY are the number of samples in DY and nN are the number of samples in DN. The Information
Gain (IG) of a split is defined as the the diff erence of the Entropy and the split entropy:

                      IG(D, DY, DN) = H(D) - H(DY, DN)                                                      (1)

The higher the information gain the better.
The Gini index of a data set is defined as G(D) = 1 - (summation from i=1 to 2) * P(ci|D)^2 and the Gini index of a split is defined as the weighted average of the Gini indices of the resulting partitions:

                      G(DY, DN) =(nY/n)G(DY) +(nN/n)G(DN)                                                   (2)

The lower the Gini index the better.
Finally, the CART measure of a split is defined as:

                      CART(DY, DN) = 2(nY/n)(nN/n)(summation from i=1 to 2)|P(ci|DY) - P(ci|DN)|:            (3)
                      
The higher the CART the better.

Note: The assignment includes two data files, train.txt and test.txt. The first consists of 100 observations to use to train your classifiers; the second has 10 to test. Each file is comma-separated, and each row contains 11 values - the first 10 are attributes (a mix of numeric and categorical translated to numeric, e.g. {T,F} = {0,1}, and the final being the true class of that observation. You will needto separate attributes and class in your load(filename) function.

(a) [10 pts.] Implement the IG(D; index; value) function according to equation 1, where D is a dataset, index is the index of an attribute and value is the split value such that the split is of the form Xi <= value. The function should return the value of the information gain.
  
(b) [10 pts.] Implement the G(D; index; value) function according to equation 2, where D is a dataset, index is the index of an attribute and value is the split value such that the split is of the form Xi <= value. The function should return the value of the gini index value.

(c) [10 pts.] Implement the CART(D; index; value) function according to equation 3, where D is a dataset, index is the index of an attribute and value is the split value such that the split is of the form Xi <= value. The function should return the value of the CART value.

(d) [20 pts.] Implement the function bestsplit(D; criterion) whichwhich takes as an input a dataset D, a string value from the set {"IG", "GINI", "CART"} which specifies a measure of interest. This function should return the best possible split for measure criterion in the form of a tuple (i, value), where i is the attribute index and value is the split value. The function should probe all possible values for each attribute and all attributes to form splits of the form Xi <= value.

(e) [10 pts] Load the training data "train.txt" provided in the assignment by implementing the function load(filename), which should return a dataset D, and find the best possible split for each of the three criteria for the data (Hint: Note that some measures need to be minimized and some maximized).

(f) [10 pts] Assume that you built a very simple DT based on the only best split for each criterion from the previous question. Show the three decision trees resulting from the best splits (draw the trees, split points and decision nodes). Assume that in each of the two leaves of the tree a decision is made to classify as the majority class.

(g) [30 pts] How many classification errors will the above three classifiers make on the testing set provided in "test.txt". A classification error occurs when a testing instance is not classified as its correct class.

Note, that you need to implement the three functions classifyIG(train, test), classifyG(train, test), classifyCART(train, test) based on the best splits you found in the previous parts. Each function should take in training data, create the best single split on a single attribute using the above defined functions, and classify each observation in the test data. The output should be a list of predicted classes for the test data (in the same order, of course). Then compare these predicted classes to the true classes of the test data.
