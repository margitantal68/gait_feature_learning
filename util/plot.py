
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.utils import create_userids, print_list
from util.normalization import normalize_rows
from sklearn import metrics
import util.settings as st
import warnings


warnings.filterwarnings("ignore")

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def plot_ROC(userid, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - user '+ userid)
    plt.legend(loc="lower right")
    plt.show()
    
def plot_histogram( values, number_of_bins = 10 ):
    plt.hist(values, bins = number_of_bins)
    plt.xlabel('Bins')
    plt.ylabel('Occurrence')
    plt.title('AUC distribution ')
    plt.show()

# precodition: OUTPUT_FIGURES folder must exists

def plot_raw_data( df, NUM_SAMPLES_PER_CLASS):
    userids = create_userids( df )
    NUM_USERS = len(userids)
    for i in range(0,NUM_USERS):
        userid = userids[i]
        # print(userid)
        user_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_data = user_data.drop(user_data.columns[-1], axis=1)
        user_array = user_data.values[0:NUM_SAMPLES_PER_CLASS,:]
        rows, cols = user_array.shape
        plt.clf()
        plt.xlabel('Time')
        plt.title("User "+str(userids[ i ])) 
        for row in range(rows):
            plt.plot(user_array[row,:])
        output_file = str(userids[ i ]) + '.png'
        print(output_file)
        plt.savefig(st.OUTPUT_FIGURES+"/"+output_file)

# Mouse dynamics dx_dy

def plot_user_dx_dy_histo( df ):
    set_style()
    userids = create_userids( df )
    NUM_USERS = len(userids)
    for i in range(0,NUM_USERS):
        userid = userids[i]
        print(userid)
        user_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_data = user_data.drop(user_data.columns[-1], axis=1)

        user_dx = user_data[ user_data.columns[range(0,128)] ]
        user_dy = user_data[ user_data.columns[range(128,256)] ]
    
        plt.clf()
        result = []
        [result.extend(el) for el in user_dx.values.tolist()]
        sns.distplot(result, norm_hist=True, color='green', bins=32)
        
        # n, bins, patches = plt.hist( result, 50 )
        plt.xlabel('Bins')
        plt.ylabel('Density')
        plt.title(' dx histogram ')
        output_file = str(userids[ i ]) + '_dx.png'
        print(output_file)
        plt.savefig(st.OUTPUT_FIGURES+"/"+output_file)

        plt.clf()
        result = []
        [result.extend(el) for el in user_dy.values.tolist()]
        ax = sns.distplot(result, norm_hist=True, color='red', bins=32)
        print(ax)
        # plt.hist( result )
        plt.xlabel('Bins')
        plt.ylabel('Density')
        plt.title(' dy histogram ')
        output_file = str(userids[ i ]) + '_dy.png'
        # print(output_file)
        plt.savefig(st.OUTPUT_FIGURES+"/"+output_file)


# Mouse dynamics
def plot_user_dt_histo( df ):
    set_style()
    userids = create_userids( df )
    NUM_USERS = len(userids)
    for i in range(0,NUM_USERS):
        userid = userids[i]
        print(userid)
        user_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_data = user_data.drop(user_data.columns[-1], axis=1)
        user_dt = user_data[ user_data.columns[range(0,128)] ]
    
        plt.clf()
        result = []
        [result.extend(el) for el in user_dt.values.tolist()]
        sns.distplot(result, norm_hist=True, color='green', bins=32)
        
        # n, bins, patches = plt.hist( result, 50 )
        plt.xlabel('Bins')
        plt.ylabel('Density')
        plt.title(str(userids[ i ]) + ' dt histogram ')
        output_file = str(userids[ i ]) + '_dt.png'
        print(output_file)
        plt.savefig(st.OUTPUT_FIGURES+"/"+output_file)

        



def plot_data_distribution( filename, plotname ):
    
    fig = plt.figure(figsize=(15, 12))

    # df = pd.read_csv(filename, header = None)
    # array = df.values
    # rows, cols = array.shape
    # L = np.ravel( array[:,0:cols-1] )
     
    # plt.hist(L, color='#3F5D7D')
    
    train = pd.read_csv(filename, header = None)
    cols = 5
    print("Average, Stdev, Min, Max")
    # loop over cols^2 vars
    for i in range(0, cols * cols):
        plt.subplot(cols, cols, i+1)
        f = plt.gca()
        # f.axes.get_yaxis().set_visible(False)
        # f.axes.set_ylim([0, train.shape[0]])

        # vals = np.size(train.iloc[:, i].unique())
        # if vals < 10:
        #     bins = vals
        # else:
        #     vals = 10

        # plt.hist(train.iloc[:, i], bins=30, color='#3F5D7D')
        mean_value = round(np.mean(train.iloc[:, i]), 2) 
        std_value = round(np.std(train.iloc[:, i]), 2)
        min_value = round(np.min(train.iloc[:, i]), 2)
        max_value = round(np.max(train.iloc[:, i]), 2)

        print(str(mean_value)+", "+ str(std_value) +", "+ str(min_value) +", "+ str(max_value) )
        plt.boxplot(train.iloc[:, i])

    plt.tight_layout()

    plt.savefig(plotname)
    plt.show()




# Plots t-SNE projections of the genuine and forged signatures
# input_csv/forgery_mcyt_1.csv
# input_csv/genuine_mcyt_1.csv
def plot_tsne_binary():
    df_genuine = pd.read_csv("input_csv/genuine_mcyt_1.csv", header =None)
    df_forgery = pd.read_csv("input_csv/forgery_mcyt_1.csv", header =None)
    
    NUM_USERS = 100
    for user in range(0, NUM_USERS):
        userlist = [ user ]
        df_user_genuine = df_genuine.loc[df_genuine[df_genuine.columns[-1]].isin(userlist)]
        df_user_forgery = df_forgery.loc[df_forgery[df_forgery.columns[-1]].isin(userlist)]

        print(df_user_genuine.shape)
        print(df_user_forgery.shape)

        G = df_user_genuine.values
        F = df_user_forgery.values

        df1 = pd.DataFrame(G[:,0:1024])
        df2 = pd.DataFrame(F[:,0:1024])

        df = pd.concat( [df1, df2] )
        

        print(df.shape)

        y = [0] * 25 + [1] * 25  
        y = LabelEncoder().fit_transform(y)
        X = df.values
        tsne = TSNE(n_components=2, init='random', random_state=41)
        X_2d = tsne.fit_transform(X, y)

    
    
    
        # target_ids = np.unique(y)
        for i in [0,1]:
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
        plt.legend(['Genuine', 'Forgery'])
    
        plt.title("User "+str(user))
        plt.show()


# Plots t-SNE projections of NUM_USERS signatures
# Shows the separability of the users based on raw data
# or other features


# converts a string to a number
# eg. u001 --> 1
def myfunc( str ):
    return (int)(str[1:])
    

#  df - dataframe, last column contains the userid
# [START_USER, STOP_USER)
# 

def plot_tsne(df, START_USER, STOP_USER, output_fig_name, plot_title):
    # df = pd.read_csv(input_name)
    rows, cols = df.shape

    # df['user'] = df['user'].apply(lambda x: myfunc(x) )
    select_classes = [ i for i in range(START_USER, STOP_USER)]
    df = df.loc[df[df.columns[-1]].isin(select_classes)]

    X = df.values
    X = X[:, 0:cols]
    y = X[:, -1]
    y = LabelEncoder().fit_transform(y)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X)

    target_ids = np.unique(y)
    fig = plt.figure()
   
    # colors = clrs = sns.color_palette('husl', n_colors=NUM_USERS) 
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'tan', 'orange', 'purple'
    for i, c, label  in zip(target_ids,colors, target_ids):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c = c, label=label)
    # plt.legend()
    # for i in target_ids:
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
  
  
    # legendstr =[]
    # for i in range(START_USER, STOP_USER):
    #     legendstr.append("USER "+str(i))
    # plt.legend( legendstr)
    
    plt.title(plot_title)
    plt.savefig( output_fig_name)
    plt.show()




def plot_scores(positive_scores, negative_scores, filename='scores.png', title='Score distribution'):
    set_style()
    plt.clf()
    df = pd.DataFrame([positive_scores, negative_scores])
    BINS = np.linspace(df.min(), df.max(), 31)
   
    sns.distplot(positive_scores, norm_hist=True, color='green', bins=31)
    sns.distplot(negative_scores, norm_hist=True, color='red', bins=31)
    # plt.legend(loc='upper left')
    
    
    plt.legend(['Genuine', 'Impostor'], loc='best')
    plt.xlabel('Score')
    plt.title(title)
    plt.show()
    plt.savefig(filename + '.png', format='png')
    plt.savefig(filename + '.eps', format='eps')
    


def plot_scores_old(positive_scores, negative_scores, filename='scores.png', title='Score distribution'):
    set_style()
    plt.clf()
    plt.hist(positive_scores, bins=100, alpha = 0.5, color = 'green',  density = True, stacked = True)
    plt.hist(negative_scores, bins=100, alpha = 0.5, color = 'orange', density = True, stacked = True)

    plt.legend(['Genuine', 'Impostor'], loc='best')
    plt.xlabel('Score')
    plt.title(title)
    plt.show()
    plt.savefig(filename, format='png')


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper", font_scale = 1.5)
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_style("ticks")
    sns.set_style("whitegrid")


# all the three files contain FPR and TPR values for ROC curves
# raw - raw data
# ae  - autoencoder data
# ee  - endtoend model data

def plot_ROCs(raw_file, ae_file, ee_file, title = 'ROC curve', outputfilename='roc.png'):
    set_style()
    raw_data = pd.read_csv(raw_file) 
    ae_data  = pd.read_csv(ae_file) 
    ee_data  = pd.read_csv(ee_file) 

    print(raw_data.shape)
    auc_raw = metrics.auc(raw_data['FPR'], raw_data['TPR'])
    auc_ae = metrics.auc(ae_data['FPR'], ae_data['TPR'])
    auc_ee = metrics.auc(ee_data['FPR'], ee_data['TPR'])

    plt.clf()
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.plot(raw_data['FPR'], raw_data['TPR'], ':', label = 'AUC_RAW = %0.2f' % auc_raw)
    plt.plot(ae_data['FPR'], ae_data['TPR'], '--', label = 'AUC_AE = %0.2f' % auc_ae)
    plt.plot(ee_data['FPR'], ee_data['TPR'], '-', label = 'AUC_EE = %0.2f' % auc_ee)

    label_raw = 'AUC raw = %0.2f' % auc_raw
    label_ae = 'AUC autoencoder = %0.2f' % auc_ae
    label_ee = 'AUC endtoend = %0.2f' % auc_ee

    # legend_str = ['RAW', 'autoencoder', 'endtoend']
    legend_str = [label_raw, label_ae, label_ee]
    plt.legend(legend_str)
    plt.show()
    plt.savefig(outputfilename + '.png', format='png')
    plt.savefig(outputfilename + '.eps', format='eps')

def plot_ROC_single(ee_file, title = 'ROC curve', outputfilename='roc.png'):
    set_style()
    ee_data  = pd.read_csv(ee_file) 
    auc_ee = metrics.auc(ee_data['FPR'], ee_data['TPR'])

    plt.clf()
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.plot(ee_data['FPR'], ee_data['TPR'], '-', label = 'AUC_EE = %0.2f' % auc_ee)

    label_ee = 'AUC  = %0.2f' % auc_ee
    legend_str = [label_ee]
    plt.legend(legend_str)
    plt.show()
    plt.savefig(outputfilename + '.png', format='png')
    plt.savefig(outputfilename + '.eps', format='eps')

# create a boxplot from a dataframe
# 
def csv2boxplot(df, columns, title, ylabel, outputfilename):
    myFig = plt.figure()
    res = df.boxplot(column=columns, return_type='axes')
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    myFig.savefig('output_png/boxplot_sapimouse.png', format = 'png')
    myFig.savefig(outputfilename + '.png', format='png')
    myFig.savefig(outputfilename + '.eps', format='eps')
    # plt.show(res)

