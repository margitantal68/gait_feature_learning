from enum import Enum

class NormalizationDirection( Enum ):
    ROWS = "rows"
    COLUMNS = "columns"
    USERS = "users-columns"
    NONE = "none"

class NormalizationType( Enum ):
    MINMAX ="minmax"
    ZSCORE ="zscore"
    NONE ="none"

class ModelType( Enum ):
    FCN = "fcn"
    RESNET = "ResNet"
    

class DataType( Enum ):
    GAIT = "gait"
    SIGNATURE = "signature"
    MOUSE = "mouse"

class AugmentationType( Enum ):
    # Circular shift
    CSHIFT ="cshift"
    # Random noise
    RND = "rnd"


#  which representation learning is used
class RepresentationType(Enum):
    AE = "autoencoder"
    EE = "endtoend"

 
DATA_TYPE = DataType.GAIT

MODEL_TYPE = ModelType.FCN

# OUTPUT_FIGURES = "OUTPUT_FIGURES"
TRAINED_MODELS_PATH = "TRAINED_MODELS"
TRAINING_CURVES_PATH ="TRAINING_CURVES"

# Init random generator
RANDOM_STATE = 11235

# Use data augmentation
AUGMENT_DATA = False

# Set verbose mode on/off
VERBOSE = True

# Model parameters
BATCH_SIZE = 16
EPOCHS = 100

# Temporary filename - used to save ROC curve data
TEMP_NAME = "temp.csv"


# Input shape GAIT
FEATURES = 128
DIMENSIONS = 3

# Aggregate consecutive blocks
AGGREGATE_BLOCK_NUM = 1
