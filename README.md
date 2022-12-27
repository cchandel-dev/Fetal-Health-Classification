# Fetal_Health_Classification
Using various Machine Learning models and amalgamating them into an Ensemble Learning model to classify fetal health and evaluate the results. 
The following algorithms were used; Random Forest, Decision Tree, KNN, SVM, Naive Bayes.

Steps to use this program are shown below.

      1. clone this repo to your local workspace
      2. in your command terminal either enter the virtual environment provided or make
          your own and install everything in requirements.txt
      3. feel free to run the EnsembleLearning.py or any of the constituent model generators 



Description of the selected forecasting problem: Using data on fetal health the goal is to forecast the health of the given fetus into three categories;
normal, suspect, or pathological. This is a classic classification problem where data is used to group the output into types as opposed to regression 
where a numerical value(s) is/are calculated.

Description of the available data: There are 21 attributes which can be used to forecast the output which is the 22nd field. 
There are 2126 samples taken in this dataset providing a healthy sample size for the classification algorithms we are looking to work with today. 
The 21 attributes are all numerical, even if they describe categorical data meaning that the dataset was already vectorized. This dataset is not normalized, 
the methods we use will have to be able to deal with unnormalized data or we will have to normalize the dataset if necessary. 
There is no missing data in the dataset. Every attribute and typically 80% of the samples will be used to train the model (train_X & train_Y).
The model will be tested using typically 20% of the samples and every attribute except the output (text_X) and compared to the true output (test_Y). 
This 80/20 split will apply to both cross-validation and hold-out as shown later on in this report.


The classification report is shown below
![classification repot](https://github.com/cchandel-dev/Fetal-Health-Classifier/blob/main/classification%20report.png)
