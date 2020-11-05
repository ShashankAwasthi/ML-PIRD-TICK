#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import catboost as ct

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
#sns.despine(top=True, right=True, left=True, bottom=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
#import catboost as ct


# ### SQL CONNECTION

# In[2]:


# execute the SP which prepares the temp tables with the data of features
# import the final temp table into dataframe
#takes 1 min


conn_dswplm = pyodbc.connect('Driver={SQL Server};'
                      'Server=CGTSAPX21012.S2.MS.Unilever.com\CGTSSQL20702P,12165;'
                      'Database=DSWPLM;'
                      'Trusted_Connection=yes;')
cursor=conn_dswplm.cursor()


# In[9]:


try:
    cursor.execute("exec SP_ML_CUPIRD_DISPLAY_TICK_Scoping")
except pyodbc.Error as err:
    print ('Error !!!!! %s' % err)


# In[3]:


#takes 3:30 mins 
# sqldf = pd.read_sql(sql="SELECT * FROM STG_ML_CUPIRD_TEMP35_ING_LIST_DISPLAY",con=conn_dswplm)


# In[4]:


# sqldf.to_excel('sql.xlsx',index=False)


# In[3]:


sqldf=pd.read_excel('sql.xlsx')


# In[4]:


sqldf.head()


# In[5]:


# isolate the sql imported df and the df used subsequently
df = sqldf.copy(deep = True)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


#d2 = datetime.datetime(2018, 6, 1)
df['PIRD_CREATE_DATE'] = pd.to_datetime(df['PIRD_CREATE_DATE']).dt.date


# In[10]:


df['Month'] = pd.to_datetime(df['PIRD_CREATE_DATE']).dt.month.map("{:02}".format)
df['Year'] = pd.to_datetime(df['PIRD_CREATE_DATE']).dt.year
df.head()


# In[11]:


df['Month_Year'] = df['Month'].map(str)+'-' + df['Year'].map(str)
df.drop(['Month'], axis=1,inplace=True)
df.drop(['Year'], axis=1,inplace=True)


# In[12]:


df_new=df.copy(deep=True)


# In[13]:


df_new.head(2)


# In[ ]:





# # FEATURE ENGINEERING

# Spliiting the GMC into 2 Columns

# In[14]:


GMC_of_PAM_new=df['GMC_of_PAM'].str.split("_",n=1,expand=True)


# In[15]:


GMC_of_PAM_new.head()


# In[16]:


df['PAM_COMP_CLASS']=GMC_of_PAM_new[0]
df['PAM_COMP_COMM']=GMC_of_PAM_new[1]


# In[17]:


df.head()


# In[18]:


df.drop(['GMC_of_PAM'], axis=1,inplace=True)


# In[19]:


print("Original shape : ",df.shape)
print("Total rows with nulls: ",df.shape[0] - df.dropna().shape[0])
#df.dropna(inplace=True)
#print("Shapes after dropping null: ",df.shape)


# In[20]:


df.isnull().sum()


# In[21]:


# sns.heatmap(df.isnull());


# In[ ]:





# In[22]:


df2 = df.copy(deep=True)


# In[23]:


df2.head(3)


# In[24]:


df2.info()


# #### Aggregation of features done based on correlation

# In[25]:


df2['TFC'] = df2['TFC_CATEGORY'].map(str)  + df2['TFC_CLASS'].map(str) + df2['TFC_SUBCLASS'].map(str) + df2['TFC_TYPE'].map(str)
df2['LANGUAGE_VAL'] = df2['LANGUAGE'].map(str) + '_'  + df2['VALIDITY_AREA'].map(str)
df2['RATING_DIV'] = df2['RATING'].map(str)  + '_' + df2['DIVISION'].map(str)
df2.drop(['LANGUAGE','VALIDITY_AREA','TFC_CATEGORY','TFC_CLASS','TFC_SUBCLASS','TFC_TYPE','RATING','DIVISION','LEGAL_DISPLAY'], axis=1,inplace=True)
#df2.drop(['BRAND'], axis=1,inplace=True)


# In[26]:


df2.LTXTFLG = df2.LTXTFLG.replace('X',1)
df2.LTXTFLG = df2.LTXTFLG.replace(' ',0)


# In[27]:


df2.LTXTFLG.unique()


# In[28]:


df2.head()


# In[29]:


df2.ING_DISPLAY = df2.ING_DISPLAY.replace('X',1)
df2.ING_DISPLAY.fillna(0,inplace=True)


# In[30]:


df2.ING_DISPLAY.value_counts()


# In[31]:


df2.head(2)


# In[32]:


df2.ALGN_DISPLAY = df2.ALGN_DISPLAY.replace('X',1)
df2.ALGN_DISPLAY.fillna(0,inplace=True)


# In[33]:


df2.ALGN_DISPLAY.value_counts()


# In[34]:


df2.head(2)


# In[ ]:





# In[35]:


# df2.to_excel('ded.xlsx')


# In[ ]:





# In[36]:


df_new=df2.copy(deep=True)


# In[37]:


df_new.head(2)


# In[ ]:





# In[38]:


d=pd.DataFrame()
d=df_new.groupby(['AUTH_GROUP','Month_Year'])['ALGN_DISPLAY'].value_counts().reset_index(name='count')#.count().reset_index(name='total')
d


# In[39]:


a=df_new.groupby(['AUTH_GROUP','Month_Year'])['ALGN_DISPLAY'].count().reset_index(name='total')

a


# In[40]:


l=pd.merge(d,a,how='inner',on=['AUTH_GROUP','Month_Year'])
l['ratio']= l['count'] /l['total']

l


# In[41]:


q=l[l['ALGN_DISPLAY']==1]
q[['AUTH_GROUP','Month_Year','total','ratio']]


# In[42]:


q2=q['Month_Year'].str.split("-",n=1,expand=True)


# In[43]:


q2.head()


# In[44]:


q['Month']=q2[0]
q['Year']=q2[1]


# In[45]:


q.head()


# In[46]:


w= q.sort_values(by=['Year','Month','AUTH_GROUP'], ascending=True).reset_index(drop =True)#.to_excel("ALGN_tick_ratio.xlsx",index=False)
w


# In[47]:


w= w[['AUTH_GROUP','ALGN_DISPLAY','Month','Year','count','total','ratio']]
w


# In[48]:


# w.to_excel("ALGN_tick_ratio.xlsx",index=False)


# In[ ]:





# # SPLITTING INTO ING,ALGN,LEGAL

# In[ ]:





# In[49]:


df_ing = df2[df2.TEXTCAT == 'ZPL_IL'].reset_index(drop=True)
df_algn = df2[df2.TEXTCAT == 'ZPL_AL'].reset_index(drop=True)
#df_legal = df2[df2.TEXTCAT == 'ZPL_LD'].reset_index(drop=True)


# In[50]:


print(df_ing.shape)
#print(df_legal.shape)
print(df_algn.shape)


# In[51]:


df3=df2.copy(deep=True)


# In[52]:


df_algn.info()


# In[53]:


df3.head(2)


# In[54]:


for x in df3.columns:
    df3[x] = pd.factorize(df3[x])[0]
    


# In[55]:


corr = df3[df3.columns].corr().abs()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr.apply(lambda x : np.round(x,2)), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True ,cmap='coolwarm')
plt.show()


# In[ ]:





# # ALGN DISPLAY TICK

# In[56]:


df_algn.head(2)


# In[57]:


df_algn2 =df_algn.copy(deep=True)


# In[58]:


df_algn2.drop(['LABELID','TEXTCAT','LTXTFLG'], axis=1,inplace=True)


# In[59]:


# df_algn2 =df_algn.copy(deep=True)


# In[60]:


for x in df_algn2.columns[df_algn2.columns != 'PIRD_CREATE_DATE']:
    df_algn2[x] = pd.factorize(df_algn2[x])[0]


# In[61]:


# df_algn.columns[df_algn.columns != 'PIRD_CREATE_DATE']


# In[62]:


df_algn2.info()


# In[63]:


#df_new =df_algn2[(df_algn2['PIRD_CREATE_DATE'] >= np.datetime64('2020-08-01')) ]#.reset_index(drop =True)
df_new =df_algn2[(df_algn2['PIRD_CREATE_DATE'] >= np.datetime64('2020-05-01')) & (df_algn2['PIRD_CREATE_DATE'] < np.datetime64('2020-06-01'))].reset_index(drop =True)


# In[64]:


df_old =df_algn2#[df_algn2['PIRD_CREATE_DATE'] < np.datetime64('2020-05-01')]#.reset_index(drop =True)


# In[65]:


#df_new.drop(['PIRD_CREATE_DATE'], axis=1,inplace=True)
df_old.drop(['PIRD_CREATE_DATE'], axis=1,inplace=True)


# In[66]:


df_old.columns


# In[67]:


df_new


# In[68]:


df_old.shape,df_new.shape


# In[69]:


corr = df_old[df_old.columns].corr().abs()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr.apply(lambda x : np.round(x,2)), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True,cmap ='coolwarm')
plt.show()


# In[70]:


corr = df_new[df_new.columns].corr().abs()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr.apply(lambda x : np.round(x,2)), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True,cmap ='coolwarm')
plt.show()


# In[71]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

# #import catboost as ct


# ## DT Model Development

# In[72]:


X_2 = df_new.drop('ALGN_DISPLAY',axis=1)
y_2 = df_new.ALGN_DISPLAY


# In[73]:


X = df_old.drop('ALGN_DISPLAY',axis=1)
y = df_old.ALGN_DISPLAY


# In[74]:


# for x in X.columns:
#     X[x] = pd.factorize(X[x])[0]
    


# In[75]:


li =[]
for i in range(0,100):
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,random_state =i)
    


# In[90]:


X_test.columns


# In[76]:


X_test.groupby('AUTH_GROUP').count()['BRAND']


# In[77]:


X_train.groupby('AUTH_GROUP').count()['BRAND']


# In[78]:


model = DecisionTreeClassifier()#criterion="entropy")#,max_depth = 40 )
model.fit(X_train,y_train)
pred = model.predict(X_test)
#print(metrics.accuracy_score(pred,y_test)*100)
li.append(metrics.accuracy_score(pred,y_test)*100)


# In[79]:


max(li)


# In[80]:


np.mean(li)


# In[81]:


min(li)


# In[82]:


print(metrics.classification_report(pred,y_test))


# In[83]:


plt.figure(figsize=(10,10))
pd.Series(model.feature_importances_,index = X.columns,).sort_values().plot(kind='barh')
plt.show()


# In[86]:


import pickle


# In[87]:


pickle.dump(model, open('model.pkl','wb'))


# In[88]:


modeling = pickle.load(open('model.pkl','rb'))


# In[89]:


modeling


# # 2020

# In[86]:


X_test.head()


# In[87]:


X_2.head()


# In[89]:


pred = model.predict(X_2)
print(metrics.accuracy_score(pred,y_2)*100)
#li.append(metrics.accuracy_score(pred,y_test)*100)


# In[ ]:


# pred = model.predict(X_test)
# print(metrics.accuracy_score(pred,y_test)*100)
# #li.append(metrics.accuracy_score(pred,y_test)*100)


# In[90]:


model.fit(X_train,y_train)
pred = model.predict(X_2)
print(metrics.accuracy_score(pred,y_2)*100)


# In[91]:


print(metrics.classification_report(pred,y_2))


# In[92]:


fig, axs = plt.subplots(ncols=2,figsize=(10, 5))

#sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1])
sns.countplot(x = df_new.ALGN_DISPLAY,data = df_new, ax=axs[0]);
sns.countplot(x = df_old.ALGN_DISPLAY,data = df_old,ax=axs[1]);


# In[93]:


# plt.figure(figsize=(20,10))
# tree.plot_tree(model,feature_names=X_train.columns,max_depth=3,fontsize=15);


# # CATBOST

# In[ ]:





# In[94]:


model2 = ct.CatBoostClassifier(cat_features=X.columns)
model2.fit(X_train,y_train)


# In[95]:


pred = model2.predict(X_test)
print(metrics.accuracy_score(pred,y_test)*100)

print(metrics.classification_report(pred,y_test))
print(X.columns)
pred2 = model2.predict(X_2)
print(metrics.accuracy_score(pred2,y_2)*100)


# In[96]:


plt.figure(figsize=(10,10))
pd.Series(model.feature_importances_,index = X.columns,).sort_values().plot(kind='barh')
plt.show()


# In[97]:


X_train


# In[98]:


# X_test.groupby ( )
X_test.groupby('AUTH_GROUP').count()


# In[99]:


df_algn_test = df_algn.loc[X_test.index]
df_algn_test['Predicted Value'] = pred
df_algn_test['Actual Value'] = y_test
df_algn_test['Accuracy Prediction'] = np.where(df_algn_test['Predicted Value']==df_algn_test['Actual Value'],1,0)


# In[100]:


df_algn_test.head(2)


# In[101]:


for x in sorted(df_algn_test['AUTH_GROUP'].unique()):
    #print(x,metrics.classification_report(df_ing_test[df_ing_test.RATING_AUTH==x]['Actual Value'],df_ing_test[df_ing_test.RATING_AUTH==x]['Predicted Value']))
    print(x,df_algn_test[df_algn_test.AUTH_GROUP==x].shape,metrics.accuracy_score(df_algn_test[df_algn_test.AUTH_GROUP==x]['Actual Value'],df_algn_test[df_algn_test.AUTH_GROUP==x]['Predicted Value']))


# In[102]:


for x in sorted(df_algn_test['AUTH_GROUP'].unique()):
    
    print(x,df_algn[df_algn.AUTH_GROUP==x].shape)


# In[103]:


for x in sorted(df_algn_test['Month_Year'].unique()):
    #print(x,metrics.classification_report(df_ing_test[df_ing_test.RATING_AUTH==x]['Actual Value'],df_ing_test[df_ing_test.RATING_AUTH==x]['Predicted Value']))
    print(x,df_algn_test[df_algn_test.Month_Year==x].shape,metrics.accuracy_score(df_algn_test[df_algn_test.Month_Year==x]['Actual Value'],df_algn_test[df_algn_test.Month_Year==x]['Predicted Value']))


# In[104]:


for x in sorted(df_algn_test['Month_Year'].unique()):
    
    print(x,df_algn[df_algn.Month_Year==x].shape)


# In[105]:


df_algn_test.head()


# In[106]:


df_algn_test.columns


# In[107]:


df_algn_test.sort_values(by=['AUTH_GROUP', 'BRAND', 'SPECIFICATION_TYPE', 'PAM_Spec_type','PAM_COMP_CLASS', 'PAM_COMP_COMM', 'TFC', 'LANGUAGE_VAL','RATING_DIV']).to_excel("aaa.xlsx",index=False)


# In[108]:


df_algn_test[(df_algn_test['AUTH_GROUP']=='ZBVG')&(df_algn_test['BRAND']=='BRBF0648')&(df_algn_test['SPECIFICATION_TYPE']=='ZCUC_TEAHB')&(df_algn_test['PAM_Spec_type']=='ZPAM_FLEXI')&(df_algn_test['LANGUAGE_VAL']=='6N_GB')&(df_algn_test['PAM_COMP_COMM']=='CUST-GMC24888')]


# #  ALL MONTHS

# In[109]:


min(list(df_algn2.PIRD_CREATE_DATE)),max(list(df_algn2.PIRD_CREATE_DATE))


# In[ ]:





# In[ ]:


df_algn2 =df_algn.copy(deep=True)


# In[ ]:


df_algn2.drop(['LABELID','TEXTCAT','LTXTFLG'], axis=1,inplace=True)


# In[ ]:


# df_algn2 =df_algn.copy(deep=True)


# In[ ]:


df_algn2.Month_Year.unique()


# In[ ]:


df_algn2.head(2)


# In[ ]:



for x in df_algn2.columns[~df_algn2.columns.isin(['PIRD_CREATE_DATE','Month_Year'])]:
    df_algn2[x] = pd.factorize(df_algn2[x])[0]


# In[ ]:


li=[]
for i in df_algn2.Month_Year.unique()[1:]:
    mon_yr=i.split(sep="-",maxsplit=-1)
    mo=mon_yr[0]
    yr=mon_yr[1]
    date= yr + '-' + mo + '-01'
    if mo=='12':
        mo1='01'
        yr=str(int(yr)+1)
    elif int(mo) >9 and int(mo) <12 :
        mo1=mon_yr[0][0] + str(int(mon_yr[0][1]) + 1)
    elif mo=='09':
        mo1=str(int(mon_yr[0][1]) + 1)
    else:
        mo1= '0'+ str(int(mon_yr[0][1]) + 1)
    date2=yr + '-' + mo1 + '-01'
    df_new =df_algn2[(df_algn2['PIRD_CREATE_DATE'] >= np.datetime64(date)) & (df_algn2['PIRD_CREATE_DATE'] < np.datetime64(date2))]
    df_old =df_algn2[df_algn2['PIRD_CREATE_DATE'] < np.datetime64(date)]
    print(date,date2,df_new.shape,df_old.shape)
    
    


# In[110]:


li=[]
for i in df_algn2.Month_Year.unique()[1:]:
    mon_yr=i.split(sep="-",maxsplit=-1)
    mo=mon_yr[0]
    yr=mon_yr[1]
    date= yr + '-' + mo + '-01'
    if mo=='12':
        mo1='01'
        yr=str(int(yr)+1)
    elif int(mo) >9 and int(mo) <12 :
        mo1=mon_yr[0][0] + str(int(mon_yr[0][1]) + 1)
    elif mo=='09':
        mo1=str(int(mon_yr[0][1]) + 1)
    else:
        mo1= '0'+ str(int(mon_yr[0][1]) + 1)
    date2=yr + '-' + mo1 + '-01'
    df_new =df_algn2[(df_algn2['PIRD_CREATE_DATE'] >= np.datetime64(date)) & (df_algn2['PIRD_CREATE_DATE'] < np.datetime64(date2))]
    df_old =df_algn2[df_algn2['PIRD_CREATE_DATE'] < np.datetime64(date)]
    
    df_new.drop(['PIRD_CREATE_DATE','Month_Year'], axis=1,inplace=True)
    df_old.drop(['PIRD_CREATE_DATE','Month_Year'], axis=1,inplace=True)

    X_2 = df_new.drop('ALGN_DISPLAY',axis=1)
    y_2 = df_new.ALGN_DISPLAY

    X = df_old.drop('ALGN_DISPLAY',axis=1)
    y = df_old.ALGN_DISPLAY
    #if X_2.shape[0]>0:
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y)
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    #print(X_2.shape)
    #print(X_test.shape)
    #print(mo,yr)
    #print(metrics.classification_report(pred,y_test))
    #print(metrics.accuracy_score(pred,y_test)*100)
    pred2 = model.predict(X_2)
    print(mo,yr,metrics.accuracy_score(pred2,y_2)*100)
    li.append(pred2)

    df_algn_test = df_algn.loc[X_2.index]
    df_algn_test['Predicted Value'] = pred2
    df_algn_test['Actual Value'] = y_2
    df_algn_test['Accuracy Prediction'] = np.where(df_algn_test['Predicted Value']==df_algn_test['Actual Value'],1,0)


    for x in sorted(df_algn_test['AUTH_GROUP'].unique()):
        #print(x,metrics.classification_report(df_ing_test[df_ing_test.RATING_AUTH==x]['Actual Value'],df_ing_test[df_ing_test.RATING_AUTH==x]['Predicted Value']))
        #print(mo,yr,df_algn_test[df_algn_test.AUTH_GROUP==x].shape,x,metrics.accuracy_score(df_algn_test[df_algn_test.AUTH_GROUP==x]['Actual Value'],df_algn_test[df_algn_test.AUTH_GROUP==x]['Predicted Value']))
        a=1


# In[ ]:


df_algn_test


# In[ ]:





# In[ ]:


df_algn2.head(2)


# In[ ]:


df_new =df_algn2[(df_algn2['PIRD_CREATE_DATE'] >= np.datetime64('2020-05-01')) & (df_algn2['PIRD_CREATE_DATE'] < np.datetime64('2020-06-01'))]
df_old =df_algn2[df_algn2['PIRD_CREATE_DATE'] < np.datetime64('2020-05-01')]

df_new.drop(['PIRD_CREATE_DATE','Month_Year'], axis=1,inplace=True)
df_old.drop(['PIRD_CREATE_DATE','Month_Year'], axis=1,inplace=True)

X_2 = df_new.drop('ALGN_DISPLAY',axis=1)
y_2 = df_new.ALGN_DISPLAY

X = df_old.drop('ALGN_DISPLAY',axis=1)
y = df_old.ALGN_DISPLAY

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y)
model = DecisionTreeClassifier()#criterion="entropy")#,max_depth = 40 )
model.fit(X_train,y_train)
pred = model.predict(X_test)

#print(metrics.classification_report(pred,y_test))
print(metrics.accuracy_score(pred,y_test)*100)

pred2 = model.predict(X_2)
print(metrics.accuracy_score(pred2,y_2)*100)
#print(metrics.classification_report(pred,y_2))

df_algn_test = df_algn.loc[X_test.index]
df_algn_test['Predicted Value'] = pred
df_algn_test['Actual Value'] = y_test
df_algn_test['Accuracy Prediction'] = np.where(df_algn_test['Predicted Value']==df_algn_test['Actual Value'],1,0)

df_algn_test.head(2)

for x in sorted(df_algn_test['AUTH_GROUP'].unique()):
    #print(x,metrics.classification_report(df_ing_test[df_ing_test.RATING_AUTH==x]['Actual Value'],df_ing_test[df_ing_test.RATING_AUTH==x]['Predicted Value']))
    print(x,metrics.accuracy_score(df_algn_test[df_algn_test.AUTH_GROUP==x]['Actual Value'],df_algn_test[df_algn_test.AUTH_GROUP==x]['Predicted Value']))


# In[ ]:


d=pd.DataFrame()
d=df_algn.groupby(['AUTH_GROUP','Month_Year'])['ALGN_DISPLAY'].value_counts().reset_index(name='count')#.count().reset_index(name='total')
d[d.Month_Year=='08-2018']


# In[ ]:


a=df_algn.groupby(['AUTH_GROUP','Month_Year'])['ALGN_DISPLAY'].count().reset_index(name='total')

a[a.Month_Year=='08-2018']


# In[ ]:


l=pd.merge(d,a,how='inner',on=['AUTH_GROUP','Month_Year'])
l['ratio']= l['count'] /l['total']
l['ratio']= l['count'] /l['total']
l[l.Month_Year=='08-2018']


# In[ ]:



q =l[if l.count==l.total and l.ALGN_DISPLAY ==0 l[l['ALGN_DISPLAY']==0] ,l[l['ALGN_DISPLAY']==1]]
#         q=l[l['ALGN_DISPLAY']==0]
# else:
#         q=l[l['ALGN_DISPLAY']==1]
# q[['AUTH_GROUP','Month_Year','total','ratio']]
# q[q.Month_Year=='08-2018']


# In[ ]:


q2=q['Month_Year'].str.split("-",n=1,expand=True)


# In[ ]:


q2.head()


# In[ ]:


q['Month']=q2[0]
q['Year']=q2[1]


# In[ ]:



q[q.Month_Year=='08-2018']


# In[ ]:


w= q.sort_values(by=['Year','Month','AUTH_GROUP'], ascending=True).reset_index(drop =True)#.to_excel("ALGN_tick_ratio.xlsx",index=False)
w


# In[ ]:


w= w[['AUTH_GROUP','ALGN_DISPLAY','Month','Year','count','total','ratio']]
w


# In[ ]:


# w.to_excel("ALGN_tick_ratio.xlsx",index=False)


# In[ ]:




