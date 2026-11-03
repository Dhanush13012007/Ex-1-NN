<H3>ENTER YOUR NAME: M.Dhanush</H3>
<H3>ENTER YOUR REGISTER NO. 212225040065</H3>
<H3>EX. NO.1</H3>
<H3>DATE:30.10.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# Importing Libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
# Read the dataset from drive
df.head()
 # Finding Missing Values
df.isnull().sum()

 # Check For Duplicates
print(df.duplicated().sum())

# Remove Unnecessary Columns
df=df.drop(['Surname', 'Geography','Gender'], axis=1) 
# Normalize the dataset
scaler=StandardScaler()                                
df=pd.DataFrame(scaler.fit_transform(df))
df.head()

# Split the dataset into input and output
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values  

 # Splitting the data for training & Testing
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)  

# X Train and Test
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)    

# Y Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)
```

## OUTPUT:

<img width="1610" height="346" alt="Screenshot 2026-11-03 161733" src="https://github.com/user-attachments/assets/5234b486-1182-420b-86fc-3ee27eada58c" />





<img width="415" height="533" alt="Screenshot 2026-11-03 161756" src="https://github.com/user-attachments/assets/aaa42796-5bdd-4ac6-8765-bdcfd2226573" />




<img width="296" height="72" alt="Screenshot 2026-11-03 161802" src="https://github.com/user-attachments/assets/946b36ed-395b-4bdd-b1b2-2bd6cad0dd7b" />



<img width="842" height="297" alt="Screenshot 2026-11-03 161811" src="https://github.com/user-attachments/assets/aa0150ea-1003-4e1b-980d-5d4f579b9476" />



<img width="638" height="361" alt="Screenshot 2026-11-03 161818" src="https://github.com/user-attachments/assets/38bbef43-b6fe-4f3b-ad96-e589f01e6edf" />



<img width="636" height="545" alt="Screenshot 2026-11-03 161824" src="https://github.com/user-attachments/assets/dba972cf-f7a2-4305-81b5-6c8fa72c5389" />



<img width="597" height="188" alt="Screenshot 2026-11-03 161829" src="https://github.com/user-attachments/assets/84248457-870c-4b15-9bf7-8590a45b34de" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


