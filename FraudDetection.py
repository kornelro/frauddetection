#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[2]:


data = pd.read_csv('creditcard.csv')


# # FRAUD DETECTIONS

# Analiza transakcji kartami kredytowymi. Cel: wykrycie oszustw.   
# Time, V1..V28, Amount, Class    
# V1..V28 zawierają ciągłe atrybuty, które są efektem przekształcenia PCA.  
# Time - sekundy od pierwszej transakcji.  
# Amount - wartość transakcji.  

# ## PRZEGLĄD DANYCH

# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


data.describe()


# ### WNIOSKI  
# Nie mamy wartości brakujących.  
# Charakter danych zgadza się z opisem.   
# Niektóre atrybuty znacznie różnią sie odchyleniem standardowym.  
# Wszystkie średnie bliskie 0, co potwierdza normalizację danych (konieczną dla PCA).

# ## WIZUALIZACJA

# In[6]:


data['Class'].value_counts().plot(kind='bar', title='Dystrybucja klas w zbiorze')


# In[7]:


data['Class'].value_counts()


# In[8]:


labels = list(data.columns)
labels.remove('Class')
labels.remove('Time')

f, axes = plt.subplots(10, 3, figsize=(20,40))
label_number = 0
for i in range(10):
    for j in range(3):
        if label_number > 28:
            break
        sns.violinplot(x='Class', y=labels[label_number], data=data, ax=axes[i,j])
        label_number += 1


# In[9]:


plt.figure(figsize=(15,8))
sns.distplot(data[data['Class']==0]['Time'], kde=False, color='blue', label='Klasa 0')
sns.distplot(data[data['Class']==1]['Time'], kde=False, color='red', label='Klasa 1')
plt.legend()


# In[10]:


data['Hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)


# In[11]:


sns.distplot(data['Hour'], kde=False, bins=24)
plt.title('Ilość wszystkich transakcji względem godzin w ciągu dnia')
plt.ylabel('Transakcje')


# In[12]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.distplot(data[data['Class']==0]['Hour'], kde=False, bins=24)
plt.title('Ilość transakcji 0 względem godzin w ciągu dnia')
plt.ylabel('Transakcje')
plt.subplot(1,2,2)
sns.distplot(data[data['Class']==1]['Hour'], kde=False, bins=24)
plt.title('Ilość transakcji 1 względem godzin w ciągu dnia')
plt.ylabel('Transakcje')


# In[13]:


hourMeanData0 = data[data['Class']==0][['Hour', 'Amount']].groupby('Hour').mean()
hourMeanData1 = data[data['Class']==1][['Hour', 'Amount']].groupby('Hour').mean()

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.bar(hourMeanData0.index, hourMeanData0['Amount'])
plt.title('Średnia wartość transakcji 0 względem godzin w ciągu dnia')
plt.ylabel('Transakcje')
plt.xlabel('Hour')
plt.subplot(1,2,2)
plt.bar(hourMeanData1.index, hourMeanData1['Amount'])
plt.title('Średnia wartość transakcji 1 względem godzin w ciągu dnia')
plt.ylabel('Transakcje')
plt.xlabel('Hour')


# In[14]:


#atrybut pokazujacy roznice wartosci transakcji od sredniej wartosci transkacji w danej godzinie
hourMeanData = data[['Hour', 'Amount']].groupby('Hour').mean()
data['HourMeanDiff'] = data.apply(lambda x: x['Amount'] - hourMeanData['Amount'][x['Hour']], axis=1)

sns.violinplot(x='Class', y='HourMeanDiff', data=data)


# In[15]:


sns.violinplot(x='Class', y='HourMeanDiff', data=data.where(data['Amount']<1000))


# In[16]:


sns.violinplot(x='Class', y='HourMeanDiff', data=data.where(data['Amount']<1000).where(data['Hour']==0))


# In[17]:


col = data['Class'].map({0:'blue', 1:'red'})
data.plot.scatter('V4', 'V11', c=col, alpha=0.4, figsize=(20,10), legend=True, title="Klasa w zależności od V4 i V11")

custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4)]


leg = plt.legend(custom_lines, [0, 1], prop={'size': 15})

for legobj in leg.legendHandles:
    legobj.set_linewidth(15.0)


# In[18]:


col = data['Class'].map({0:'blue', 1:'red'})
data.plot.scatter('V12', 'V14', c=col, alpha=0.4, figsize=(20,10), legend=True, title="Klasa w zależności od V12 i V14")

custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4)]


leg = plt.legend(custom_lines, [0, 1], prop={'size': 15})

for legobj in leg.legendHandles:
    legobj.set_linewidth(15.0)


# In[19]:


corr = data.corr()
corr.style.background_gradient().set_precision(2)


# ### WNIOSKI
# 
# - Mamy oczywiście doczynienia z bardzo niezrównoważonym zbiorem, na co trzeba będzie zwrócić szczególną uwage przy modelowaniu  
# - Nie ma atrybutów, które bardzo wyraźnie oddzielają klasy, ale atrybuty V4, V11, V12 czy V14 wyglądają na bardziej znaczące. 
# - W przypadku oszustw nie zdażają się tak wysokie kwoty, jak w przypadku transakcji legalnych. Być może transakcje na wysokie kwoty są bardziej obserwowane i dlatego ciężej jest na takich oszukiwać.  
# - Sam atrybut Time niewiele daje, ale jego przekształcenie na godziny pozwala wyciągać bardziej sensowne wnioski. Tutaj zakładam, że pierwsza transakcja w zbiorze była wykonana na samym początku dnia.  
# - W godzinach 0-10 wykonuje się mniej legalnych transakcji, co nie jest tak dobrze widoczne w przypadku oszustw, które są wykonywane również w nocy. Transakcje wykonywane w nocy mają większą szansę na bycie oszustwem.  
# - W transkacjach oszukanych nie zdarzają się bardzo wysokie kwoty, jednak często są to średnio kwoty wyższe niż w transkacjach legalnych. Dlatego utworzyłem atrybut pokazujący różnicę wartości transakcji od średniej wartosci transakcji dla danej godziny. Wydaje się on być znaczący dla transakcji o niskich wartościach i jeżeli jeszcze weźmiemy pod uwagę godzinę. W przypadku tego artrybutu trzeba uważać, aby "nie wróżyć z fusów" ;), jego wartość będę liczył używająć średnich tylko ze znanego zbioru testowego.  
# - Przykładowe scatter ploty względem V4, V11, V12, V14 pokazują, że klasy można nawet dobrze odserparować. Myślę, że z takimi podziałami powinny poradzić sobie drzewa.  
# - Między atrybutami nie występują silne korelacje.  

# ## MODELOWANIE

# Miara, której użyję do oceny modelu to F1score, ponieważ chcemy mieć pewność, że oznaczone przez nas oszustwa to oszustwa (precision), ale też znaleźć jak najwięcej oszustw (recall).  
# 
# Do modelowania pozbędę się atrybutu time, ale dodam atrybuty Hour, HourMeanDiff oraz spróbuję też z atrybutem binarnym, gdzie 1 oznacza noc.
# 
# Do zbioru testowego wyznaczę 100 oszustw i 400 transakcji legalnych
# 
# Pierwszą próbą będzie regresja logistyczna, na zrównoważonym zbiorze - usunięta zostanie znaczna część transakcji normalnych. Tutaj niestety stracimy dużo informacji.
# 
# Sróbuję też z lasem lososwym, na którym każde k drzew będzie uczone na zbiorze składającym się zawsze z tego samego zbioru oszustw i losowo wybranego zbioru transakcji normalnych (w różnych bardziej zrównoważonych proporcjach, np 1:1, 2:1, 3:1). W tym podejściu będziemy mogli wykorzystać więcej informacji ze zbioru danych.

# ### Przygotowanie zbiorów

# In[20]:


data = pd.read_csv('creditcard.csv')
data['Hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
data['Night'] = data['Hour'].apply(lambda x: 1 if x>=0 and x<=10 else 0)
data = data.drop(labels=['Time'],axis=1)


# In[21]:


testData = data[data['Class']==1].sample(100, random_state=1111)
testData = pd.concat([testData, data[data['Class']==0].sample(400, random_state=1111)])
trainData = data.drop(testData.index)


# In[22]:


print(testData.shape[0] + trainData.shape[0] == data.shape[0])


# In[23]:


hourMeanData = trainData[['Hour', 'Amount']].groupby('Hour').mean()
trainData['HourMeanDiff'] = trainData.apply(lambda x: x['Amount'] - hourMeanData['Amount'][x['Hour']], axis=1)
testData['HourMeanDiff'] = testData.apply(lambda x: x['Amount'] - hourMeanData['Amount'][x['Hour']], axis=1)


# In[24]:


trainData.head()


# In[25]:


testData.head()


# ### Regresja logistyczna

# In[26]:


data = trainData.copy()
regressionTrainData = data[data['Class']==1]
regressionTrainData = pd.concat([regressionTrainData, data[data['Class']==0].sample(regressionTrainData.shape[0],
                                                                                             random_state=1111)])


# In[27]:


clf = LogisticRegression(random_state=1111)
clf.fit(regressionTrainData.drop(labels=['Class'],axis=1), regressionTrainData['Class'])


# In[28]:


y_pred = clf.predict(testData.drop(labels=['Class'],axis=1))
print('f1score: ',f1_score(testData['Class'], y_pred))

matrix = confusion_matrix(testData['Class'], y_pred)
df_cm = pd.DataFrame(matrix, index = [i for i in [0,1]], columns = [i for i in [0,1]])

plt.figure()
plt.title("Confusion matrix")
sns.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')


# ### Las losowy

# In[29]:


def getTrainData(proportion):
    data = trainData.copy()
    toReturn = data[data['Class']==1]
    toReturn = pd.concat([toReturn, data[data['Class']==0].sample(toReturn.shape[0] * proportion)])
    return toReturn


# In[30]:


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


# In[31]:


#pierwszy las
classifier_forest = RandomForestClassifier(n_estimators=100)
rfTrainData = getTrainData(1)
classifier_forest.fit(rfTrainData.drop(labels=['Class'],axis=1), rfTrainData['Class'])

y_pred = classifier_forest.predict(testData.drop(labels=['Class'],axis=1))
print('f1score: ',f1_score(testData['Class'], y_pred))

matrix = confusion_matrix(testData['Class'], y_pred)
df_cm = pd.DataFrame(matrix, index = [i for i in [0,1]], columns = [i for i in [0,1]])

plt.figure()
plt.title("Confusion matrix")
sns.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')


# In[32]:


#dodawanie kolejnych
proportions = [1, 2, 3, 1, 2]

for proportion in proportions: 
    classifier_forest_next = RandomForestClassifier(n_estimators=100)
    rfTrainData = getTrainData(proportion)
    classifier_forest_next.fit(rfTrainData.drop(labels=['Class'],axis=1), rfTrainData['Class'])
    classifier_forest = combine_rfs(classifier_forest, classifier_forest_next)


# In[33]:


y_pred = classifier_forest.predict(testData.drop(labels=['Class'],axis=1))
print('f1score: ',f1_score(testData['Class'], y_pred))

matrix = confusion_matrix(testData['Class'], y_pred)
df_cm = pd.DataFrame(matrix, index = [i for i in [0,1]], columns = [i for i in [0,1]])

plt.figure()
plt.title("Confusion matrix")
sns.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')


# In[34]:


features_importance = pd.DataFrame(data={'feature': rfTrainData.drop(labels=['Class'],axis=1).columns,
                                         'importance': classifier_forest.feature_importances_})
print(features_importance.sort_values('importance', ascending=False))


# ### WNIOSKI

# - Regresja logistyczna dała trochę lepsze wyniki niż pierwszy las losowy ze 100 drzewami.
# - Wynik F1score poprawił się gdy do lasu dołożyliśmy drzewa trenowane na kolejnych porcjach danych.
# - Tu trzeba zwrócić uwagę, że poprawiła się wartość recall, ale trochę spadła wartość precision.  
# - Las wydaje się lepszym modelem, bo został wytrenowany na większej próbce danych niż regresja logistyczna.  
# - Las nie jest dobrze interpretowalnym modelem, ale możemy skorzystać z feature_importance. W rankingu ważności wysoko zostały wskazane m.in. atrybuty, które wskazałem po wizualizacji. Z dodanych przeze mnie atrybutów najważniejszy okazał się HourMeanDiff, ale nie zajął on znacznie wysokiego miejsca.

# In[ ]:




