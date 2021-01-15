import pandas as pd
import seaborn as sns

from util.utils import create_userids, print_list
from util.autoencoder import train_autoencoder, generate_autoencoder_samples

from util.augment_data import get_augmented_dataset
from util.oneclass import evaluate_authentication,evaluate_authentication_cross_day
from util.normalization  import normalize_rows
from util.plot import plot_raw_data

import util.settings as st
from util.model import train_model, evaluate_model, get_model_output_features
from util.autoencoder import get_autoencoder_output_features

from util.settings import AugmentationType, RepresentationType, DataType
from util.plot import plot_ROCs, plot_tsne, set_style


data_type = DataType.GAIT

def generate_model_name(representation_type, fcn_filters, augm= False, augm_type=AugmentationType.RND):
    # print(representation_type)
    model_name ="gait_"+representation_type.value+"_fcn"+str(fcn_filters)+"_"
    if augm == True:
        model_name = model_name + augm_type.value
    model_name = model_name + '.h5'
    return model_name

# ------------------------------------------- VERIFICATION --------------------------------------------------------

# training: IDNet
# evaluation ZJU_Gait 
# 
def evaluate_autoencoder_authentication(training = False, fcn_filters = 128, augm = False, augm_type=AugmentationType.RND,   verbose = False, filename ='results/autoencoder_scores.csv'):
    representation_type=RepresentationType.AE
    model_name = generate_model_name(representation_type, fcn_filters = fcn_filters, augm = augm, augm_type = augm_type)
    if training  == True:
        df_idnet = pd.read_csv("input_csv_gait/idnet_raw.csv", header = None) 
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)
        train_autoencoder(df_idnet, model_name=model_name, fcn_filters=fcn_filters)

    encoder_name = "encoder_"+model_name
    
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 = get_autoencoder_output_features( df1, encoder_name )
    features2 = get_autoencoder_output_features( df2, encoder_name )

    roc_data_filename = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc1, eer1 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc2, eer2 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)

    scores_filename ='results/auc_eer_users_' + str(representation_type.value) + '_' + str(data_type.value) + '.csv'
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(scores_filename, index=False)


# training: IDNet
# evaluation ZJU_Gait 
# 
def evaluate_endtoend_authentication(training = False, fcn_filters = 128, augm = True, augm_type = AugmentationType.RND, verbose = False, filename ='results/endtoend_scores.csv'):
    representation_type = RepresentationType.AE
    model_name = generate_model_name(representation_type, fcn_filters = fcn_filters, augm = augm, augm_type = augm_type)
    if training  == True:
        df_idnet = pd.read_csv("input_csv_gait/idnet_raw.csv", header = None)  
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)      
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        train_model(df_idnet, model_name, fcn_filters, representation_learning=True)

    
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 = get_model_output_features( df1, model_name )
    features2 = get_model_output_features( df2, model_name )

    roc_data_filename = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc1, eer1 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc2, eer2 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)

    scores_filename ='results/auc_eer_users_' + str(representation_type.value) + '_' + str(data_type.value) + '.csv'
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(scores_filename, index=False)

def evaluate_raw_authentication(verbose = False, filename ='results/raw_scores.csv'): 
    representation_type=RepresentationType.RAW
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 =  df1
    features2 =  df2

    roc_data_filename = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc1, eer1 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc2, eer2 = evaluate_authentication( features1, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)
    roc_data_filename = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, data_type, representation_type, verbose = verbose, roc_data = True, roc_data_filename = roc_data_filename)

    scores_filename ='results/auc_eer_users_' + str(representation_type.value) + '_' + str(data_type.value) + '.csv'
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(scores_filename, index=False)



# evaluates raw, autoencoder and endtoend models for session1
def evaluate_authentication_session1(augm = False, verbose = False, augm_type = AugmentationType.RND):
    # RAW
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)

    representation_type = RepresentationType.RAW
    roc_data_filename1 = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( df1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename1)
    
    # AE
    fcn_filters = 128
    representation_type = RepresentationType.AE
    model_name = generate_model_name(representation_type=RepresentationType.AE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    encoder_name = "encoder_"+model_name
    features1 = get_autoencoder_output_features( df1, encoder_name )
    roc_data_filename2 = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( features1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename2)

    
    # EE
    representation_type = RepresentationType.EE
    model_name = generate_model_name(representation_type=RepresentationType.EE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    features1 = get_model_output_features( df1, model_name )
    roc_data_filename3 = 'results/roc_session1_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( features1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename3)

    # plot ROC
    roc_filename = 'output_png/roc_session1_' + str(data_type.value)
    title = 'Same-day evaluation: session1'
    plot_ROCs(roc_data_filename1, roc_data_filename2, roc_data_filename3, title = title, outputfilename= roc_filename)


# evaluates raw, autoencoder and endtoend models for session2
def evaluate_authentication_session2(augm = False, verbose = False, augm_type = AugmentationType.RND):
    # RAW
    df1 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)

    representation_type = RepresentationType.RAW
    roc_data_filename1 = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( df1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename1)
    
    # AE
    representation_type = RepresentationType.AE
    fcn_filters = 128
    model_name = generate_model_name(representation_type=RepresentationType.AE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    encoder_name = "encoder_"+model_name
    features1 = get_autoencoder_output_features( df1, encoder_name )
    roc_data_filename2 = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( features1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename2)

    
    # EE
    representation_type = RepresentationType.EE
    model_name = generate_model_name(representation_type=RepresentationType.EE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    features1 = get_model_output_features( df1, model_name )
    roc_data_filename3 = 'results/roc_session2_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication( features1, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename3)

    # plot ROC
    roc_filename = 'Same-day evaluation: session2'
    title = 'ROC curve ' + str(data_type.value)
    plot_ROCs(roc_data_filename1, roc_data_filename2, roc_data_filename3, title = title, outputfilename= roc_filename)



 # evaluates raw, autoencoder and endtoend models cross-session
def evaluate_authentication_cross_session(augm = False, verbose= False, augm_type = AugmentationType.RND):
    # RAW
    representation_type = RepresentationType.RAW
    df1 = pd.read_csv("input_csv_gait/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv_gait/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)
    
    roc_data_filename1 = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication_cross_day( df1, df2, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename1)
    

    # AE
    representation_type = RepresentationType.AE
    fcn_filters = 128
    model_name = generate_model_name(representation_type=RepresentationType.AE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    encoder_name = "encoder_"+model_name
    features1 = get_autoencoder_output_features( df1, encoder_name )
    features2 = get_autoencoder_output_features( df2, encoder_name )

    roc_data_filename2 = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication_cross_day( features1, features2, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename2)
    

    # EE
    representation_type = RepresentationType.EE
    model_name = generate_model_name(representation_type=RepresentationType.EE, fcn_filters = fcn_filters, augm = augm, augm_type=augm_type)
    features1 = get_model_output_features( df1, model_name )
    features2 = get_model_output_features( df2, model_name )

    roc_data_filename3 = 'results/roc_cross_session_' + str(representation_type.value) + '_' + str(data_type.value)+'.csv'
    evaluate_authentication_cross_day( features1, features2, data_type, representation_type, verbose, roc_data = True, roc_data_filename = roc_data_filename3)
    
    # plot ROC
    roc_filename = 'output_png/roc_cross_session_' + str(data_type.value)
    title = 'Cross-day evaluation'
    plot_ROCs(roc_data_filename1, roc_data_filename2, roc_data_filename3, title = title, outputfilename= roc_filename)

# ------------------------------------------- MAIN --------------------------------------------------------
    
if __name__ == "__main__":
    #  Verification + ROC curves
    # evaluate_authentication_session1(augm = False)
    evaluate_authentication_session2(augm = False)
    # evaluate_authentication_cross_session(augm = False)
    
    # Verification
    # evaluate_raw_authentication(verbose=False)
    # evaluate_autoencoder_authentication(training = False, fcn_filters = 128, augm = False, augm_type = AugmentationType.RND, verbose = False)
    # evaluate_endtoend_authentication(training = False,  fcn_filters = 128, augm = True, augm_type = AugmentationType.RND, verbose = False)

# ------------------------------------------- END MAIN --------------------------------------------------------

