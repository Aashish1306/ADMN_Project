#!/usr/bin/env python
# coding: utf-8

# In[1]:





import streamlit as st




import numpy as np





st.title("Bank_Personal_Loan")





import pandas as pd





my_data = pd.read_csv("C:/Users/ASUS/Documents/Course documents/Sem 3/ADMN 5016/Bank_Personal_Loan_Modelling.csv")
my_data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","Personal_Loan","SecuritiesAccount","CDAccount","Online","CreditCard"]











my_data[my_data['Experience'] < 0]['Experience'].count()





my_dataExp = my_data.loc[my_data['Experience'] >0]
negExp = my_data.Experience < 0
column_name = 'Experience'
my_data_list = my_data.loc[negExp]['ID'].tolist()





negExp.value_counts()





for id in my_data_list:
    age = my_data.loc[np.where(my_data['ID']==id)]["Age"].tolist()[0]
    education = my_data.loc[np.where(my_data['ID']==id)]["Education"].tolist()[0]
    df_filtered = my_dataExp[(my_dataExp.Age == age) & (my_dataExp.Education == education)]
    exp = df_filtered['Experience'].median()
    my_data.loc[my_data.loc[np.where(my_data['ID']==id)].index, 'Experience'] = exp





my_data[my_data['Experience'] < 0]['Experience'].count()







# In[3]:


from sklearn import metrics


# In[4]:


from sklearn import *


# In[5]:


from sklearn.metrics import accuracy_score


# In[6]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.metrics import accuracy_score


# In[9]:


data=my_data.drop(['ID','ZIPCode','Experience'], axis =1 )
data.head(10)


# In[10]:


data1=data[['Age','Income','Family','CCAvg','Education','Mortgage','SecuritiesAccount','CDAccount','Online','CreditCard','Personal_Loan']]


# In[11]:


data1.head(10)


# In[12]:


data1["Personal_Loan"].value_counts(normalize=True)


# In[13]:


st.dataframe(data1,width = 1500)


# In[14]:


st.write(data1.describe())


# In[ ]:





# In[ ]:





# In[15]:


array = data1.values
X = array[:,0:10] 
Y = array[:,10]  
test_size = 0.30 
seed = 15 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
type(X_train)


# In[16]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print('Accuracy:',model_score)
print('confusion_matrix:')
print(metrics.confusion_matrix(y_test, y_predict))
A=model_score  


# In[17]:


def get_user_input():
    Age = st.sidebar.slider('Age',0,100,15)
    Income = st.sidebar.slider('Income',0,225,100)
    Family = st.sidebar.slider('Family',0,5,2)
    CCAvg = st.sidebar.slider('CCAvg',0,10,5)
    Education = st.sidebar.slider('Education',0,3,1)
    Mortage = st.sidebar.slider('Mortgage',0,650,100)
    SecuritiesAccount = st.sidebar.selectbox('SecuritiesAccount',('0','1'))
    CDAccount = st.sidebar.selectbox('CDAccount',('0','1'))
    Online = st.sidebar.selectbox('Online',('0','1'))
    CreditCard = st.sidebar.selectbox('CreditCard',('0','1'))
                                      
    
    user_data = {'Age':Age,
                'Income': Income,
                 'Family': Family,
                 'CCAvg': CCAvg,
                 'Education' : Education,
                 'Mortgage' : Mortage,
                 'SecuritiesAccount' : SecuritiesAccount,
                 'CDAccount' : CDAccount,
                 'Online' : Online,
                 'CreditCard' : CreditCard}
    features = pd.DataFrame(user_data, index = [0])
    return features



# In[ ]:



    


# In[18]:


data1.describe()


# In[ ]:





# In[19]:


user_input = get_user_input()


# In[20]:


user_input.count()


# In[21]:


st.subheader('User Input:')
st.write(user_input)


# In[22]:



print(user_input)


# In[198]:


st.subheader('Personal Loan:')
prediction = model.predict(user_input)
st.write(prediction)


# In[209]:


st.subheader('Accuracy of Model')
acc = accuracy_score(y_test,model.predict(X_test))
st.write(acc)


# In[ ]:




