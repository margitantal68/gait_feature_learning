# gait_feature_learning
A comparison of supervised and unsupervised feature learning methods in accelerometer-based gait verification


## Used datasets
* Dataset used for evaluations
   * [ZJU-GaitAcc] - http://www.ytzhang.net/datasets/zju-gaitacc/
      * session 0 - 22 subjects
      * session 1 - 153 subjects
      * session2 - 153 subjects
   
* Dataset used for feature learning
   *[IDNet] - http://signet.dei.unipd.it/research/human-sensing/
      * 50 subjects with various number of sessions
      * Resampled at 100 Hz


## Segmentation

* FRAME-based: length = 128 samples (Sampling frequency 100 Hz)



## Features
   * RAW - use raw accelerometer data as features - 3 x 128 = 384 (ax - ay - az) 
   * SUPERVISED feature extraction
      * Convolutional end-to-end model (FCN) trained on IDNet
	
   * UNSUPERVISED feature extraction - autoencoders
      * Fully Convolutional (FCN) autoencoder trained on IDNet
      
## Verification - based on a single gait segment (FRAME)
   * OneClass SVM (OCSVM) for each user
   * Two protocols:
      * SAME-DAY: using data from a single session  
      * CROSS-DAY: training - session 1, testing - session 2
            
## Code
  * The main_gait.py python file contains the necessary code to run an experiment.
  * The TRAINED_MODELS folder contains the end-to-end models as well as the autoencoders trained in different settings (with or without augmentation)
  * The plots folder contains the source codes necessary to create the figures in the paper.
  * The util folder contains the following:
    * augment_data.py - functions used for data augmentation
    * autoencoder.py - code for training and evaluating autoencoders
    * classification.py - code for user identification (classification)
    * fcn.py - Fully Convolutional end-to-end model
    * model.py - code for training the end-to-end model
    * normalization.py - functions for data normalization
    * oneclass.py - code for user verification
    * plot.py - different utility plots
    * utils.py - utility functions
    * settings.py - enums and constants

## Results
   * 
   * Scores 
      * Scores produced by autoencoder features
       ![Scores produced by autoencoder features](https://github.com/margitantal68/gait_feature_learning/blob/main/results/sameday_ae_scores_orig.png?raw=true)
      * Scores produced by end-to-end features  
      ![Scores produced by end-to-end features](https://github.com/margitantal68/gait_feature_learning/blob/main/results/sameday_ee_scores_orig.png)


   * ROC curves using global thresholds
      * Same-day - session 1
      ![ROC - same day - session1](https://github.com/margitantal68/gait_feature_learning/blob/main/results/roc_session1.png?raw=true)
      * Same day - session 2
      ![ROC - same day - session2](https://github.com/margitantal68/gait_feature_learning/blob/main/results/roc_session2.png?raw=true)
      * Cross day
      ![ROC - same day - session2](https://github.com/margitantal68/gait_feature_learning/blob/main/results/roc_cross_session.png?raw=true)
   
   
  