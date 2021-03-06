import pandas as pd
from util.utils import create_userids, print_list
from util.autoencoder import train_autoencoder, generate_autoencoder_samples

from util.augment_data import get_augmented_dataset
from util.oneclass import evaluate_authentication, evaluate_authentication_skilledforgeries, evaluate_authentication_cross_day
from util.normalization  import normalize_rows
from util.classification import evaluate_identification_CV
from util.plot import plot_raw_data

import util.settings as st
from util.model import train_model, evaluate_model, get_model_output_features
from util.autoencoder import get_autoencoder_output_features
from util.utils import create_userids,  create_userid_dictionary, create_bigram_100_csv
from util.classification import evaluate_identification_CV, evaluate_identification_Train_Test

from util.settings import AugmentationType, RepresentationType

from util.plot import plot_ROCs




def generate_model_name(representation_type, fcn_filters, augm= False, augm_type=AugmentationType.RND):
    # print(representation_type)
    model_name ="gait_"+representation_type.value+"_fcn"+str(fcn_filters)+"_"
    if augm == True:
        model_name = model_name + augm_type.value
    model_name = model_name + '.h5'
    return model_name


# training: IDNet
# evaluation ZJU_Gait
# 
def evaluate_autoencoder_identification(training = False, num_folds = 10, fcn_filters = 128, augm = False, augm_type=AugmentationType.RND):
    model_name = generate_model_name(representation_type=RepresentationType.AE, fcn_filters = fcn_filters, augm = augm, augm_type = augm_type)

    if training  == True:
        df_idnet = pd.read_csv("input_csv_gait/idnet_raw.csv", header = None) 
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)
        train_autoencoder(df_idnet, model_name=model_name, fcn_filters=fcn_filters )

    encoder_name = "encoder_" + model_name
    
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)


    features1 = get_autoencoder_output_features( df1, encoder_name )
    features2 = get_autoencoder_output_features( df2, encoder_name )

    print("\nSession 1")
    evaluate_identification_CV(features1, num_folds = num_folds)
    print("\nSession 2")
    evaluate_identification_CV(features2, num_folds = num_folds)
     print("\nCross-Session")
    evaluate_identification_Train_Test(features1, features2)


# training: IDNet
# evaluation ZJU_Gait
# 
def evaluate_endtoend_identification(training = False, num_folds = 10, fcn_filters = 128, augm = False, augm_type=AugmentationType.RND):
    model_name = generate_model_name(representation_type=RepresentationType.EE, fcn_filters = fcn_filters, augm = augm, augm_type = augm_type)
    
    if training  == True:
        df_idnet = pd.read_csv("input_csv_gait/idnet_raw.csv", header = None)  
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)      
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        train_model(df_idnet, model_name, fcn_filters = fcn_filters, representation_learning=True)
    
    
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 = get_model_output_features( df1, model_name)
    features2 = get_model_output_features( df2, model_name)

    print("\nSession 1")
    evaluate_identification_CV(features1, num_folds = num_folds)
    print("\nSession 2")
    evaluate_identification_CV(features2, num_folds = num_folds)
     print("\nCross-Session")
    evaluate_identification_Train_Test(features1, features2)



def evaluate_raw_identification(num_folds = 10):
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    print("\nSession 1")
    evaluate_identification_CV(df1, num_folds = num_folds)
    print("\nSession 2")
    evaluate_identification_CV(df2, num_folds = num_folds)
    print("\nCross-Session")
    evaluate_identification_Train_Test(df1, df2)


# ------------------------------------------- MAIN --------------------------------------------------------


def main():
    
    print("**********   AUTOENCODER FEATURES  **********")
    evaluate_autoencoder_identification(training = False)
    print("**********   END-TO-END FEATURES  **********")
    evaluate_endtoend_identification(training = False)
    print("**********   RAW FEATURES  **********")
    evaluate_raw_identification()


if __name__ == "__main__":
    main()

# ------------------------------------------- END MAIN --------------------------------------------------------



