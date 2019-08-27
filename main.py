# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:06:14 2019

@author: Xavier Littée
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


def ask_parameters_to_user(Question):
    local_test=False
    while local_test==False:
        try:    
            answer=float(input(Question+'  '))
            local_test=True
        except Exception as e:
            print(str(e))
    return answer
    
def ask_yes_or_no_question(Question):
    local_test=False
    while local_test==False:
        try:    
            answer=input(Question+ '  (y/n) ')
            if answer.capitalize()=='Y' or answer.capitalize()=='N':
                local_test=True            
        except Exception as e:
            print(str(e))
        return answer.capitalize()
    
    
print("The training of the model has begun, it should take  around  4 minutes")
def train_model():
       
    
    start_time=time.time()
    os.chdir('N:/Python')
    
    path='betclic_datascience_test_churn.csv'
    
    Data=pd.read_csv(path,';')
    
    
    #Data=Data.head(10000)
    
    
    #-----------DATA PRE PROCESSING-----------------------
    Data=Data.rename(columns = {"regsitration_date": "registration_date"}) 
    
    Data=Data.replace([np.inf, -np.inf, 'NaT'], np.nan)
    
    Data["date_of_birth"]=Data.date_of_birth.apply(lambda x: float(x))
    
    Data["registration_date"]=pd.to_datetime(Data.registration_date, format='%Y-%m-%d')
    Data["registration_date_ord"]=Data.registration_date.apply(lambda x: x.toordinal())
    Data["registration_month"]=Data.registration_date.apply(lambda x: x.month)
    Data["registration_month_decal"]=Data.registration_date.apply(lambda x: x.month+6 if (x.month<7) else x.month-6)
    
    Data["transaction_date"]=pd.to_datetime(Data.transaction_date, format='%Y-%m-%d')
    Data["transaction_date_ord"]=Data.transaction_date.apply(lambda x: x.toordinal())
    Data["transaction_month"]=Data.transaction_date.apply(lambda x: x.month)
    Data["transaction_month_decal"]=Data.transaction_date.apply(lambda x: x.month+6 if (x.month<7) else x.month-6)
    
    
    Data["transaction_date_ord_2"]=np.nan
    Data["customer_key_2"]=np.nan
    
    Data=Data.sort_values(['customer_key','transaction_date_ord'],ascending=True).reset_index(drop=True)
    
    #DataCopy, Data= Data, Data.head(10000)
    
    Data.transaction_date_ord_2.loc[range(0,len(Data)-1)]=Data.transaction_date_ord.loc[range(1,len(Data))].values
    Data.customer_key_2.loc[range(0,len(Data)-1)]=Data.loc[range(1,len(Data)),'customer_key'].values
    
    Data['nb_inactive_days']=Data.transaction_date_ord_2-Data.transaction_date_ord
    Data.loc[Data.customer_key!=Data.customer_key_2,'nb_inactive_days']=np.nan
    
    
    Data["mean_inactive_days"]=Data.groupby(['customer_key']).nb_inactive_days.transform('mean')
    Data["max_inactive_days"]=Data.groupby(['customer_key']).nb_inactive_days.transform('max')
    Data.bet_amount.loc[Data["bet_amount"]==0]=np.nan 
    Data["mean_bet_amount"]=Data.groupby(['customer_key']).bet_amount.transform('mean')
    Data.bet_nb.loc[Data["bet_nb"]==0]=np.nan 
    Data["mean_bet_nb"]=Data.groupby(['customer_key']).bet_nb.transform('mean')
    
    Data["relative_PnL"]=Data._1/Data.bet_amount
    Data.relative_PnL.loc[Data["relative_PnL"]==0]=np.nan 
    Data["mean_PnL"]=Data.groupby(['customer_key']).relative_PnL.transform('mean')
    
    Data.deposit_amount.loc[Data["deposit_amount"]==0]=np.nan 
    Data["mean_deposit_amount"]=Data.groupby(['customer_key']).deposit_amount.transform('mean')
    Data.deposit_nb.loc[Data["deposit_nb"]==0]=np.nan 
    Data["mean_deposit_nb"]=Data.groupby(['customer_key']).deposit_nb.transform('mean')
    Data["nb_bet_per_month_per_client"]=Data.groupby(['customer_key','transaction_month']).bet_nb.transform('sum')
    Data["nb_bet_per_month_per_client_decal"]=Data.groupby(['customer_key','transaction_month_decal']).bet_nb.transform('sum')
    
    # max(tree.labels.items(), key=operator.itemgetter(1))[0]
    Data["most_active_month"]=Data.loc[Data.groupby(['customer_key']).nb_bet_per_month_per_client.transform('idxmax'),'transaction_month'].reset_index(drop=True)
    Data["most_active_month_decal"]=Data.loc[Data.groupby(['customer_key']).nb_bet_per_month_per_client_decal.transform('idxmax'),'transaction_month_decal'].reset_index(drop=True)
    
    
    #%%-----------------------Features----------------------------
    xData=Data.groupby(['customer_key']).nb_inactive_days.max().reset_index()
    xData["mean_inactive_days"]=Data.groupby(['customer_key']).mean_inactive_days.first().reset_index(drop=True)
    xData["date_of_birth"]=Data.groupby(['customer_key']).date_of_birth.first().reset_index(drop=True)
    xData["acquisition_channel"]=Data.groupby(['customer_key']).acquisition_channel_id.first().reset_index(drop=True)
    xData["gender"]=Data.groupby(['customer_key']).gender.first().reset_index(drop=True).map({'M':'0','F':'1'})
    xData["most_active_month"]=Data.groupby(['customer_key']).most_active_month.first().reset_index(drop=True)
    xData["most_active_month_decal"]=Data.groupby(['customer_key']).most_active_month_decal.first().reset_index(drop=True)
    xData["mean_bet_amount"]=Data.groupby(['customer_key']).mean_bet_amount.first().reset_index(drop=True)
    
    xData["mean_bet_nb"]=Data.groupby(['customer_key']).mean_bet_nb.first().reset_index(drop=True)
    xData["mean_PnL"]=Data.groupby(['customer_key']).mean_PnL.first().reset_index(drop=True)
    
    xData["mean_deposit_amount"]=Data.groupby(['customer_key']).mean_deposit_amount.first().reset_index(drop=True)
    xData["mean_deposit_nb"]=Data.groupby(['customer_key']).mean_deposit_nb.first().reset_index(drop=True)
    xData["date_of_birth"]=Data.groupby(['customer_key']).date_of_birth.first().reset_index(drop=True)
    
    xData["activity_per_month"]=(Data.groupby(["customer_key"]).transaction_date.count()/((Data.groupby(["customer_key"]).transaction_date_ord.max()-Data.groupby(["customer_key"]).registration_date_ord.min())/30)).reset_index(drop=True)
    activity_per_month=Data.groupby(["customer_key"]).transaction_date.count()/((Data.groupby(["customer_key"]).transaction_date_ord.max()-Data.groupby(["customer_key"]).registration_date_ord.min())/30)
    
    xData=xData.rename(columns={'nb_inactive_days': 'max_inactive_days'})
    acquisition_channels=Data.groupby(["acquisition_channel_id"]).customer_key.count()
    categories=Data.sort_values(['customer_key','transaction_date_ord','betclic_customer_segmentation'],ascending=True).reset_index(drop=True).head(100)
    
    #xData.acquisition_channel=xData.acquisition_channel.map({10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 16:'F', 17:'G', 18:'H', 20:'I', 24:'J', 25:'K', 26:'L'})
    #xData.acquisition_channel=xData.acquisition_channel.map({10:7, 11:8, 12:9, 13:10, 14:11, 16:12, 17:13, 18:14, 20:15, 24:16, 25:17, 26:18})
    dummies=pd.get_dummies(xData.acquisition_channel)
    #xData=xData.join(dummies.loc[16])
    #xData=xData.join(dummies)
    #del xData["acquisition_channel"]
    
    del xData["mean_inactive_days"]
    
    
    #    xData.date_of_birth.loc[np.where(xData.date_of_birth=='NaT')]=np.nan
    
    
    
    #xData.mean_deposit_amount.fillna(xData.mean_deposit_amount.mean(),inplace=True)
    #xData.mean_deposit_nb.fillna(xData.mean_deposit_nb.mean(),inplace=True)
    #xData.max_inactive_days.fillna(xData.max_inactive_days.mean(),inplace=True)
    
    xData=xData.reset_index(drop=True)
    xData=xData.set_index('customer_key')
    
    xData=xData.drop(['activity_per_month',"most_active_month_decal","gender"],axis=1)
    #xData=xData.drop(['activity_per_month'],axis=1)
    
    
    Churners=xData["max_inactive_days"]>=90
    del xData["max_inactive_days"]
    
    
    for col in xData.columns:
        xData[col].fillna(xData[col].mean(),inplace=True)
        
        
    yData=np.zeros(len(xData))
    yData[Churners]=1
    
    #------------------------------------------------------------------------------
    
    X_train,X_test,y_train,y_test=train_test_split(xData,yData,train_size=0.8,test_size=0.2,random_state=6)
    
    # Scale the feature data so it has mean = 0 and standard deviation = 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    
    
    ##----------UP SAMPLING------------------------------------
    smote = SMOTE(ratio='minority')
    X_train, y_train = smote.fit_sample(X_train, y_train)
    X_test, y_test = smote.fit_sample(X_test, y_test)
    
    ##-----------------------------------------------------
    
    model=LogisticRegression(solver='lbfgs')
    model=RandomForestClassifier(n_estimators=100)
    #model=DecisionTreeClassifier(max_depth=5)
    
    
    
    model.fit(X_train,y_train)
    
    # Score the model on the train data
    print(model.score(X_train,y_train))
    
    # Score the model on the test data
    labels=y_test
    guesses=model.predict(X_test)
    print(model.score(X_test,labels))
    
    
    probas=model.predict_proba(X_test)
    #coefs=pd.DataFrame(model.coef_)
    #coefs_rank=abs(coefs).rank(axis=1)
    
    prob_y_2 = model.predict_proba(X_test)
    prob_y_2 = [p[1] for p in prob_y_2]
    
    print( roc_auc_score(labels, prob_y_2) )
    
    #print(coefs)
    #print(guesses)
    #print(probas)
    
    print(accuracy_score(labels,guesses))
    print(recall_score(labels,guesses))
    print(precision_score(labels,guesses))
    print(f1_score(labels,guesses))
    
    
    print(time.time()-start_time)
    
    return model
     #%%       
  
def user_interface(model):
    #    USER INTERFACE
    label_debut=True
    while label_debut==True:
        
        birth=ask_parameters_to_user("Please enter customer's birth year")
        
        acquisition_channel_id=ask_parameters_to_user("Please enter customer's acquisition channel")
                
        most_active_month=ask_parameters_to_user("Please enter customer's most active month (int from 1 to 12)")   
        
        mean_bet_amount=ask_parameters_to_user("Please enter customer's mean BET AMOUNT (float)")
        
        mean_bet_nb=ask_parameters_to_user("Please enter customer's mean bet number (float)")
           
        mean_relative_PnL=ask_parameters_to_user("Please enter customer's mean PnL (float)")
        
        mean_deposit_amount=ask_parameters_to_user("Please enter customer's mean DEPOSIT AMOUNT (float)")
        
        mean_deposit_nb=ask_parameters_to_user("Please enter customer's mean deposit number (float)")
        
        
        test_input=[[birth, acquisition_channel_id, most_active_month,  mean_bet_amount, mean_bet_nb, mean_relative_PnL, mean_deposit_amount, mean_deposit_nb]]
        
    #    test_input=[[1987.0,	24,	4	,0.1	,1.0	,0.853,	35.16535904203834	,1.1839]]
        
        scaler.transform(test_input)
        
        print(model.predict(test_input))
        print(model.predict(scaler.transform(test_input)))
        print(model.predict_proba(scaler.transform(test_input)))
        
        label_other_try=True
        while label_other_try==True:
            other_try=ask_yes_or_no_question('Do you want to try with another customer?')
            if other_try=='N':
                label_debut=False
                
                close_app=ask_yes_or_no_question('Do you want to close the app?' )
                if close_app=='Y':
                    label_other_try=False
                else:
                    label_other_try=True
            else:
                label_debut=True
                label_other_try=False
            
            
    print('Good Bye!')
    
if __name__== "__main__":
  trained_model=train_model()
  user_interface(trained_model)
