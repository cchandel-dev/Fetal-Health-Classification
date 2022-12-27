# importing utility modules
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, classification_report


# importing voting classifier
from sklearn.ensemble import VotingClassifier


#read data into dataframe
data = pd.read_csv ('Different Models/fetal_health.txt', sep ='\t')
#Examine the first few rows 
print(data.head())


#split data into train and test sets
Y = data.fetal_health
X = data.drop('fetal_health',axis=1)
train_X, X_test, train_Y, test_Y = train_test_split(X,Y, test_size = 0.2)


# initializing all the model objects with default parameters
# Loading model to compare the results
model_1 = pickle.load(open('Different Models/pickled_models/DecisionTreesModel.pkl','rb'))
model_2 = pickle.load(open('Different Models/pickled_models/KNNModel.pkl','rb'))
model_3 = pickle.load(open('Different Models/pickled_models/NaiveBayesModel.pkl','rb'))
model_4 = pickle.load(open('Different Models/pickled_models/RandomForestModel.pkl','rb'))
model_5 = pickle.load(open('Different Models/pickled_models/SVMmodel.pkl','rb'))

# Making the final model using voting classifier
final_model = VotingClassifier(
	estimators=[('DecisionTrees', model_1), ('KNN', model_2), ('NaiveBayes', model_3), ('RandomForest', model_4), ('SVM', model_5)], voting='hard')

# training all the model on the train dataset
final_model.fit(train_X, train_Y)

# predicting the output on the test dataset
pred_final = final_model.predict(X_test)

# printing log loss between actual and predicted value 
print(classification_report(test_Y, pred_final))
