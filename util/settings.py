
from enum import Enum


class NormalizationDirection( Enum ):
    ROWS = "rows"
    COLUMNS = "columns"
    ALL = "rows&columns together"
    USERS = "users-columns"
    NONE = "none"

class NormalizationType( Enum ):
    MINMAX ="minmax"
    ZSCORE ="zscore"
    NONE ="none"

class ModelType( Enum ):
    FCN = "fcn"
    RESNET = "ResNet"
    # nem mukodik
    ENCODER ="encoder" 
    # nagyon gyenge
    MLP = "mlp"
    # nem mukodik
    MCDCNN ="mcdcnn"
    TLENET = "tlenet"
    CNN ="cnn"

class DataType( Enum ):
    GAIT = "gait"
    SIGNATURE = "signature"
    MOUSE = "mouse"
    KEYSTROKE = "keystroke"

class AugmentationType( Enum ):
    # Circular shift
    CSHIFT ="cshift"
    # Random noise
    RND = "rnd"

class EvaluationDatasetSignature(Enum):
    MCYT = "mcyt"
    MOBISIG ="mobisig"
    SVC = "svc"

# what kind of data use for training
# used for SIGNATURE
class TrainingType(Enum):
    GEN = "genuine"
    # treat forgeries as the same class --> noisy data
    GEN_FOR_N  = "genuine_forgery_N"
    # treat forgeries as a separate class 
    GEN_FOR_2N = "genuine_forgery_2N"


#  which representation learning is used
class RepresentationType(Enum):
    RAW ="rawdata"
    AE = "autoencoder"
    EE = "endtoend"


EVAL_DATASET = EvaluationDatasetSignature.MCYT
# 
TRAIN_DATASET = EvaluationDatasetSignature.MOBISIG
TRAINING_TYPE = TrainingType.GEN

DATA_TYPE = DataType.SIGNATURE

MODEL_TYPE = ModelType.FCN

OUTPUT_FIGURES = "output_png"
TRAINED_MODELS_PATH = "TRAINED_MODELS"
SAVED_MODELS_PATH = "SAVED_MODELS"
TRAINING_CURVES_PATH ="TRAINING_CURVES"

# Init random generator
RANDOM_STATE = 11235

# Model name
# MODEL_NAME = DATA_TYPE.value+"_"+MODEL_TYPE.value+".hdf5"

# Update weights
UPDATE_WEIGHTS = False

# Use data augmentation
AUGMENT_DATA = False

# Old model name whose weights will be updated
# and saved using MODEL_NAME
OLD_MODEL_NAME = "base_gait_idnet_"+MODEL_TYPE.value+".hdf5"


# Set verbose mode on/off
# VERBOSE = True

# Model parameters
BATCH_SIZE = 16
EPOCHS = 100

# Temporary filename - used to save ROC curve data
TEMP_NAME = "scores.csv"

# CNN model Input shape GCJ
# Unigrams
# FEATURES = 67
# Bigrams
# FEATURES = 6241
# DIMENSIONS = 1

MODEL_PARAMS = {
    "mouse": {
        "FEATURES": 128,
        "DIMENSIONS": 2
    },
    "keystroke": {
        "FEATURES": 31,
        "DIMENSIONS": 1
    },
    "signature": {
        "FEATURES": 512,
        "DIMENSIONS": 3
    },
    "gait": {
        "FEATURES": 128,
        "DIMENSIONS": 3
    }
}



# CNN model Input shape GAIT
FEATURES = 128
DIMENSIONS = 3




# Aggregate consecutive blocks
AGGREGATE_BLOCK_NUM = 1


# Scores & ROC settings

# Create score distribution plots for evaluations
SCORES = True

# Apply score normalization
SCORE_NORMALIZATION = True


