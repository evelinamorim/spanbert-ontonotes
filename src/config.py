class Config:
    MODEL_NAME = "SpanBERT/spanbert-base-cased"
    MAX_TRAIN_LEN = 128
    MAX_EVAL_LEN = 1024
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    EPOCHS = 3
    DATA_DIR = '../data/jsonlines_spanbert/'
    MIN_NUM_MENTIONS = 2

    MAX_NUM_SPEAKERS = 20 # copied from https://github.com/mandarjoshi90/coref
    SINGLE_EXAMPLE = 1  # copied from https://github.com/mandarjoshi90/coref
    GENRES = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"]
    MAX_SPAN_WIDTH = 30 # copied from https://github.com/mandarjoshi90/coref
    DROPOUT_RATE = .3  # copied from https://github.com/mandarjoshi90/coref
    FFNN_SIZE = 1000   # copied from https://github.com/mandarjoshi90/coref
    FFNN_DEPTH = 1   # copied from https://github.com/mandarjoshi90/coref
    USE_PRIOR = True  # copied from https://github.com/mandarjoshi90/coref

    FEATURE_SIZE = 20 # copied from https://github.com/mandarjoshi90/coref
    USE_FEATURES = True # copied from https://github.com/mandarjoshi90/coref
    MODEL_HEADS = True  # copied from https://github.com/mandarjoshi90/coref
    TOP_SPAN_RATIO = 0.4   # copied from https://github.com/mandarjoshi90/coref
    MAX_TOP_ANTECEDENTS = 50  # copied from https://github.com/mandarjoshi90/coref
    # Add other configuration parameters here
