import pandas as pd
import numpy as np
import util.normalization
import matplotlib.pyplot as plt  

from util.utils import create_userids
from util.plot import plot_ROC, plot_scores
from random import  uniform

# from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from sklearn import metrics
from util.settings import AGGREGATE_BLOCK_NUM, TEMP_NAME, DATA_TYPE, SCORES, SCORE_NORMALIZATION
from util.settings import RepresentationType, DataType
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score

def calculate_EER(y, scores):
# calculating EER of Top-S detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
#        scores = float vector

    # Calculating EER
    fpr,tpr, _ = metrics.roc_curve(y,scores,pos_label=1)
    fnr = 1-tpr
    # EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)
   
    return EER 

def compute_AUC(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def compute_AUC_EER(positive_scores, negative_scores):  
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    # plot_ROC(fpr, tpr)
    roc_auc = metrics.auc(fpr, tpr)
    fnr = 1-tpr   
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    return roc_auc, EER, fpr, tpr



def evaluate_authentication( df, data_type, representation_type, verbose = False, roc_data = False, roc_data_filename = TEMP_NAME):
    print(df.shape)
    userids = create_userids( df )
    NUM_USERS = len(userids)
    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df.loc[ df.iloc[:, -1].isin([userid]) ]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
 
        num_samples = user_array.shape[0]
        train_samples = (int)(num_samples * 0.66) + 1
        test_samples = num_samples - train_samples
        if (verbose == True):
            print(str(userid)+". #train_samples: "+str(train_samples)+"\t#test_samples: "+ str(test_samples))
        user_train = user_array[0:train_samples,:]
        user_test = user_array[train_samples:num_samples,:]
     
        other_users_data = df.loc[~df.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        other_users_array = other_users_data.values   
        
        clf = OneClassSVM(gamma='scale')
        clf.fit(user_train)
 
        positive_scores = clf.score_samples(user_test)
        negative_scores =  clf.score_samples(other_users_array)   
        
        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        auc, eer,_,_ = compute_AUC_EER(y_pred_positive, y_pred_negative)
        
        if SCORE_NORMALIZATION == True:
            positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)
        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        if  verbose == True:
            print(str(userid)+", "+ str(auc)+", "+str(eer)+"\n" )
         
        auc_list.append(auc)
        eer_list.append(eer) 
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    
    print("#positives: "+str(len(global_positive_scores)))
    print("#negatives: "+str(len(global_negative_scores)))

    global_auc, global_eer, fpr, tpr = compute_AUC_EER(global_positive_scores, global_negative_scores)
    
    filename = 'output_png/scores_'+ str(data_type.value)+ '_' + str(representation_type.value) 
    if SCORES == True:
        # ****************************************************************************************
        plot_scores(global_positive_scores, global_negative_scores, filename, title='Scores distribution')
        # ****************************************************************************************

    if( roc_data == True ):
        dict = {'FPR': fpr, 'TPR': tpr}
        df = pd.DataFrame(dict) 
        df.to_csv(roc_data_filename, index=False)

        words = roc_data_filename.split('/')
        auc_eer_data_filename = words[0] +'/auc_eer_' + words[ 1 ]
        dict = {'AUC': auc_list, 'EER': eer_list}
        df = pd.DataFrame(dict) 
        df.to_csv(auc_eer_data_filename, index=False)
        
    print("Global AUC: "+str(global_auc))
    print("Global EER: "+str(global_eer))
    return auc_list, eer_list

def evaluate_authentication_train_test( df_train, df_test, data_type, representation_type, verbose = False, roc_data = False, roc_data_filename = TEMP_NAME):
    print("Training: "+str(df_train.shape))
    print("Testing: "+str(df_test.shape))
    userids = create_userids( df_train )
    NUM_USERS = len(userids)
    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df_train.loc[ df_train.iloc[:, -1].isin([userid]) ]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
        # train_samples = user_array.shape[0]
        

        user_test_data = df_test.loc[ df_test.iloc[:, -1].isin([userid]) ]
        user_test_data = user_test_data.drop(user_test_data.columns[-1], axis=1)
        # test_samples = user_test_data.shape[0]

        other_users_data = df_test.loc[~df_test.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        # other_users_array = other_users_data.values   
        

        # if (verbose == True):
        # print(str(userid)+". #train_samples: "+str(train_samples)+"\t#positive test_samples: "+ str(test_samples))

        clf = OneClassSVM(gamma='scale')
        clf.fit(user_train_data)
 
        positive_scores = clf.score_samples(user_test_data)
        negative_scores =  clf.score_samples(other_users_data)   
        
        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        auc, eer,_,_ = compute_AUC_EER(y_pred_positive, y_pred_negative)
        # auc, eer = compute_AUC_EER(positive_scores, negative_scores )

        if SCORE_NORMALIZATION == True:
            positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)

        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        if  verbose == True:
            print(str(userid)+", "+ str(auc)+", "+str(eer) )
         
        auc_list.append(auc)
        eer_list.append(eer) 
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    
    print("#positives: "+str(len(global_positive_scores)))
    print("#negatives: "+str(len(global_negative_scores)))


    global_auc, global_eer, fpr, tpr = compute_AUC_EER(global_positive_scores, global_negative_scores)
    
    
    filename = 'output_png/scores_'+ str(data_type.value)+ '_' + str(representation_type.value)
    if SCORES == True:
        # ****************************************************************************************
        plot_scores(global_positive_scores, global_negative_scores, filename, title='Scores distribution')
        # ****************************************************************************************

    if( roc_data == True ):
        dict = {'FPR': fpr, 'TPR': tpr}
        df = pd.DataFrame(dict) 
        df.to_csv(roc_data_filename, index=False)
        
    print(data_type.value + " Global AUC: "+str(global_auc))
    print(data_type.value + " Global EER: "+str(global_eer))
    return auc_list, eer_list



# Used only for signature authentication
# df_genuine - genuine data
# df_forgery - forgeries
def evaluate_authentication_skilledforgeries( df_genuine, df_forgery, data_type, representation_type, verbose = False, roc_data = False, roc_data_filename = TEMP_NAME):
    print("Genuine shape: "+str(df_genuine.shape))
    print("Forgery shape: "+str(df_forgery.shape))
    print(df_forgery.shape)
    userids = create_userids( df_genuine )
    NUM_USERS = len(userids)
    
    global_positive_scores = list()
    global_negative_scores = list()
    auc_list = list()
    eer_list = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_genuine_data = df_genuine.loc[df_genuine.iloc[:, -1].isin([userid])]
        user_forgery_data = df_forgery.loc[df_forgery.iloc[:, -1].isin([userid])]
      
        user_genuine_data = user_genuine_data.drop(user_genuine_data.columns[-1], axis=1)
        user_genuine_array = user_genuine_data.values
 
        num_samples = user_genuine_array.shape[0]
        train_samples = (int)(num_samples * 0.66)
        test_samples = num_samples - train_samples
        # MCYT
        # train_samples = 15
        # test_samples = 10

        user_genuine_train = user_genuine_array[0:train_samples,:]
        user_genuine_test = user_genuine_array[train_samples:num_samples,:]
     
        user_forgery_data =  user_forgery_data.drop(user_forgery_data.columns[-1], axis=1) 
        user_forgery_array = user_forgery_data.values

        clf = OneClassSVM(gamma='scale')
        clf.fit(user_genuine_train)
 
        positive_scores = clf.score_samples(user_genuine_test)
        negative_scores =  clf.score_samples(user_forgery_array)   
        auc, eer,_,_ = compute_AUC_EER(positive_scores, negative_scores )
 
        if SCORE_NORMALIZATION == True:
            positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)
        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)
        
        if  verbose == True:
            print(str(userid)+": "+ str(auc)+", "+str(eer) )
        auc_list.append(auc)
        eer_list.append(eer)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
  

    global_auc, global_eer, fpr, tpr = compute_AUC_EER(global_positive_scores, global_negative_scores)
    
    filename = 'output_png/scores_'+ str(data_type.value)+ '_' + str(representation_type.value)
    if SCORES == True:
        # ****************************************************************************************
        plot_scores(global_positive_scores, global_negative_scores, filename, title='Scores distribution')
        # ****************************************************************************************

    if( roc_data == True ):
        dict = {'FPR': fpr, 'TPR': tpr}
        df = pd.DataFrame(dict) 
        df.to_csv(roc_data_filename, index=False)

    print("Global AUC: "+str(global_auc))
    print("Global EER: "+str(global_eer))

# Used only for Gait authentication
# df1 - ZJU_Gait_session1
# df2 - ZJU_Gait_session2

def evaluate_authentication_cross_day( df1, df2, data_type, representation_type, verbose = False, roc_data = False, roc_data_filename = TEMP_NAME ):
    print("Session 1 shape: "+str(df1.shape))
    print("Session 2 shape: "+str(df2.shape))
        
    userids = create_userids( df1 )
    NUM_USERS = len(userids)
    
    global_positive_scores = list()
    global_negative_scores = list()
    auc_list = list()
    eer_list = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]

        user_session1_data = df1.loc[df1.iloc[:, -1].isin([userid])]
        user_session2_data = df2.loc[df2.iloc[:, -1].isin([userid])]
      
        user_session1_data = user_session1_data.drop(user_session1_data.columns[-1], axis=1)
        user_session1_array = user_session1_data.values
 
        # positive test data
        user_session2_data =  user_session2_data.drop(user_session2_data.columns[-1], axis=1) 
        user_session2_array = user_session2_data.values

        # negative test data
        other_users_session2_data = df2.loc[~df2.iloc[:, -1].isin([userid])]
        other_users_session2_data = other_users_session2_data.drop(other_users_session2_data.columns[-1], axis=1)
        other_users_session2_array = other_users_session2_data.values   
        
        clf = OneClassSVM(gamma='scale')
        clf.fit(user_session1_array)
 
        positive_scores = clf.score_samples(user_session2_array)
        negative_scores =  clf.score_samples(other_users_session2_array)   

        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        auc, eer, _, _ = compute_AUC_EER(y_pred_positive, y_pred_negative)

        
        # auc, eer = compute_AUC_EER(positive_scores, negative_scores )
        if SCORE_NORMALIZATION == True:
            positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)

        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        

        if verbose == True:
            print(str(userid)+": "+ str(auc)+", "+str(eer) )
        auc_list.append(auc)
        eer_list.append(eer)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )

    
    global_auc, global_eer, fpr, tpr = compute_AUC_EER(global_positive_scores, global_negative_scores)
    
    filename = 'output_png/scores_'+ str(data_type.value)+ '_' + str(representation_type.value) 
    if SCORES == True:
        # ****************************************************************************************
        plot_scores(global_positive_scores, global_negative_scores, filename, title='Scores distribution')
        # ****************************************************************************************

    if( roc_data == True ):
        dict = {'FPR': fpr, 'TPR': tpr}
        df = pd.DataFrame(dict) 
        df.to_csv(roc_data_filename, index=False)

    print("Global AUC: "+str(global_auc))
    print("Global EER: "+str(global_eer))
    return auc_list, eer_list



def score_normalization(positive_scores, negative_scores):
    scores = [positive_scores, negative_scores ]
    scores_df = pd.DataFrame( scores )

    # ZSCORE normalization
    mean = scores_df.mean()
    std = scores_df.std()
    min_score =  mean - 2 * std
    max_score = mean + 2 * std

    # MIN_MAX normalization
    # min_score = scores_df.min()
    # max_score = scores_df.max()

    min_score = min_score[ 0 ]
    max_score = max_score[ 0 ]

    positive_scores = [(x - min_score)/ (max_score - min_score ) for x in positive_scores] 
    positive_scores = [(uniform(0.0, 0.05) if x < 0 else  x) for x in positive_scores ]
    positive_scores = [ ( uniform(0.95, 1.0) if x > 1 else  x) for x in positive_scores ]

    negative_scores = [(x - min_score)/ (max_score - min_score )for x in negative_scores] 
    negative_scores = [ uniform(0.0, 0.05) if x < 0 else  x for x in negative_scores ]
    negative_scores = [ uniform(0.95, 1.0) if x > 1 else  x for x in negative_scores ]


    return positive_scores, negative_scores




