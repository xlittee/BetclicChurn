# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:06:14 2019

@author: Xavier Littée
"""

import numpy as np
import pandas as pd
import warnings
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

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
    
    

def train_model():
    
    start_time=time.time()
    print("The training of the model has begun, it should take  about 3 minutes to complete")
        
    path='betclic_datascience_test_churn.csv'
    
    Data=pd.read_csv(path,';')
    
    #-----------DATA PRE PROCESSING-----------------------
    Data=Data.rename(columns = {"regsitration_date": "registration_date"}) 
    
    Data=Data.replace([np.inf, -np.inf, 'NaT'], np.nan)
    
    Data["date_of_birth"]=Data.date_of_birth.apply(lambda x: float(x))
    
    Data["registration_date"]=pd.to_datetime(Data.registration_date, format='%Y-%m-%d')
    Data["registration_date_ord"]=Data.registration_date.apply(lambda x: x.toordinal())
    Data["registration_month"]=Data.registration_date.apply(lambda x: x.month)
    
    Data["transaction_date"]=pd.to_datetime(Data.transaction_date, format='%Y-%m-%d')
    Data["transaction_date_ord"]=Data.transaction_date.apply(lambda x: x.toordinal())
    Data["transaction_month"]=Data.transaction_date.apply(lambda x: x.month)    
    
    Data["transaction_date_ord_2"]=np.nan
    Data["customer_key_2"]=np.nan
    
    Data=Data.sort_values(['customer_key','transaction_date_ord'],ascending=True).reset_index(drop=True)
    
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
    
    Data["most_active_month"]=Data.loc[Data.groupby(['customer_key']).nb_bet_per_month_per_client.transform('idxmax'),'transaction_month'].reset_index(drop=True)
    
    
    #%%-----------------------Features----------------------------
    xData=Data.groupby(['customer_key']).nb_inactive_days.max().reset_index()
#    xData["mean_inactive_days"]=Data.groupby(['customer_key']).mean_inactive_days.first().reset_index(drop=True)
    xData["date_of_birth"]=Data.groupby(['customer_key']).date_of_birth.first().reset_index(drop=True)
    xData["acquisition_channel"]=Data.groupby(['customer_key']).acquisition_channel_id.first().reset_index(drop=True)
    xData["gender"]=Data.groupby(['customer_key']).gender.first().reset_index(drop=True).map({'M':0,'F':1})
    
    xData["most_active_month"]=Data.groupby(['customer_key']).most_active_month.first().reset_index(drop=True)
    xData["registration_month"]=Data.groupby(['customer_key']).registration_month.first().reset_index(drop=True)
    #convert most_active_month and registration_month into dummy variables
    xData["most_active_month"]=xData.most_active_month.map({1:'01_act', 2:'02_act', 3:'03_act', 4:'04_act',5:'05_act', 6:'06_act', 7:'07_act', 8:'08_act', 9:'09_act', 10:'10_act', 11:'11_act', 12:'12_act'})
    xData["registration_month"]=xData.registration_month.map({1:'01_reg', 2:'02_reg', 3:'03_reg', 4:'04_reg',5:'05_reg', 6:'06_reg', 7:'07_reg', 8:'08_reg', 9:'09_reg', 10:'10_reg', 11:'11_reg', 12:'12_reg'})
    dummiesM=pd.get_dummies(xData.most_active_month)
    xData=xData.join(dummiesM)       
    dummiesM=pd.get_dummies(xData.registration_month)
    xData=xData.join(dummiesM)       
       
    
    xData["mean_bet_amount"]=Data.groupby(['customer_key']).mean_bet_amount.first().reset_index(drop=True)
    
    xData["mean_bet_nb"]=Data.groupby(['customer_key']).mean_bet_nb.first().reset_index(drop=True)
    xData["mean_PnL"]=Data.groupby(['customer_key']).mean_PnL.first().reset_index(drop=True)
    
    xData["mean_deposit_amount"]=Data.groupby(['customer_key']).mean_deposit_amount.first().reset_index(drop=True)
    xData["mean_deposit_nb"]=Data.groupby(['customer_key']).mean_deposit_nb.first().reset_index(drop=True)
    xData["date_of_birth"]=Data.groupby(['customer_key']).date_of_birth.first().reset_index(drop=True)
    
    xData["activity_per_month"]=(Data.groupby(["customer_key"]).transaction_date.count()/((Data.groupby(["customer_key"]).transaction_date_ord.max()-Data.groupby(["customer_key"]).registration_date_ord.min())/30)).reset_index(drop=True)
    
    xData=xData.rename(columns={'nb_inactive_days': 'max_inactive_days'})
    
    #convert acquisition_channel into dummy variables and keep the most important ones
    dummies=pd.get_dummies(xData.acquisition_channel)
    xData=xData.join(dummies[[13,24,25]])
    
    
    xData=xData.reset_index(drop=True)
    xData=xData.set_index('customer_key')
       
    # Labelling
    Churners=xData["max_inactive_days"]>=90
    yData=np.zeros(len(xData))
    yData[Churners]=1    
    
    #Cleaning Data
    xData=xData.drop(['activity_per_month','most_active_month','registration_month',"acquisition_channel","max_inactive_days"],axis=1)    
    
    for col in xData.columns:
        xData[col].fillna(xData[col].mean(),inplace=True)
        
        
    
    #----------UP SAMPLING------------------------------------
    smote = SMOTE(ratio='minority', random_state=123)
    xData_U, yData_U = smote.fit_sample(xData, yData)
    #------------------------------------------------------------------------------
    
    X_train,X_test,y_train,y_test=train_test_split(xData_U,yData_U,train_size=0.8,test_size=0.2,random_state=6)
    
#    # Scale the feature data if Logistic Regression
#    scaler = StandardScaler()
#    X_train = scaler.fit_transform(X_train)
#    X_test= scaler.transform(X_test)
#    
    
#    model=LogisticRegression(solver='lbfgs')
    model=RandomForestClassifier(n_estimators=100, random_state=1)
    
        
    model.fit(X_train,y_train)
    
    print('The model is a Random Forest of 100 trees:')
    print('------------------------------------------')
    # Score the model on the train data
    print("Training Accuracy : " + str(model.score(X_train,y_train)))
    
    # Score the model on the test data
    labels=y_test
    guesses=model.predict(X_test)
    print("Test Accuracy     : " + str(accuracy_score(labels,guesses)))
    
    
    prob_y_2 = model.predict_proba(X_test)
    prob_y_2 = [p[1] for p in prob_y_2]    
    print("AUC-ROC           : " + str(roc_auc_score(labels, prob_y_2) ))
    
    print("Recall            : " + str(recall_score(labels,guesses)))
    print("Precision         : " + str(precision_score(labels,guesses)))
    print("F1 Score          : " + str(f1_score(labels,guesses)))
    
    print('------------------------------------------')
    print('------------------------------------------')
    
    print(time.time()-start_time)
    
    return model
     #%%       
  
def user_interface(model):
    #    USER INTERFACE
    label_debut=True
    while label_debut==True:
        
        birth=ask_parameters_to_user("Please enter customer's birth year")
        
        gender_str=input("Please enter customer's gender (M/F)")
                        
               
        most_active_month=ask_parameters_to_user("Please enter customer's most active month (int from 1 to 12)")   
        
        registration_month=ask_parameters_to_user("Please enter customer's registration month (int from 1 to 12)")   
        
        mean_bet_amount=ask_parameters_to_user("Please enter customer's mean BET AMOUNT (float)")
        
        mean_bet_nb=ask_parameters_to_user("Please enter customer's mean bet number (float)")
           
        mean_relative_PnL=ask_parameters_to_user("Please enter customer's mean PnL (float)")
        
        mean_deposit_amount=ask_parameters_to_user("Please enter customer's mean DEPOSIT AMOUNT (float)")
        
        mean_deposit_nb=ask_parameters_to_user("Please enter customer's mean deposit number (float)")
        
        acquisition_channel_id=ask_parameters_to_user("Please enter customer's acquisition channel")
        
        acq_13=0.0
        acq_24=0.0
        acq_25=0.0
        act_month=np.zeros([1,12])
        reg_month=np.zeros([1,12])
        act_month[0,int(most_active_month)-1]=1.0
        reg_month[0,int(registration_month)-1]=1.0
        
        if gender_str.capitalize()=='F':
            gender=1.0
        else:
            gender=0.0
            
        if acquisition_channel_id==13: 
            acq_13=1
        elif acquisition_channel_id==24: 
            acq_24=1
        elif acquisition_channel_id==25: 
            acq_25=1
                    
        test_input=[[1996.8	,0,	0,	0,	0	,0	,0	,0	,0	,0,	0	,1	,0	,0	,0	,0.803439	,0	,0	,0	,0,	0,	0.196561	,0,	0,	0	,0,	7.5259	,1.58034	,0.32797	,10,	1,	0,	1	,0]]
        test_input=[[1990.0,	1.0	,0,	0,	0,	1,	0	,0	,0,	0,	0,	0,	0,	0	,0,	0,	0	,1	,0,	0	,0	,0	,0,	0,	0	,0,	0.1	,1.0	,0.853	,35.16535904203834,	1.1839732116613593	,0	,1,	0]]

        test_input=[[birth,gender] + list(act_month[0]) + list(reg_month[0])+[ mean_bet_amount, mean_bet_nb, mean_relative_PnL, mean_deposit_amount, mean_deposit_nb,acq_13,acq_24,acq_25]]
       
        
        classification=int(model.predict(test_input)[0])
        if model.predict(test_input)[0]==0:
            print("This client as been classified as a NON-CHURNER with the probability of :" + str(round(model.predict_proba(test_input)[0][classification] *100)) +'%')
        else:
            print("This client as been classified as a CHURNER with the probability of :" + str(round(model.predict_proba(test_input)[0][classification] *100)) +'%')
       
        
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
#%%    
if __name__== "__main__":
  trained_model=train_model()
  user_interface(trained_model)
