import os

class Config():

    ''' set train/test files and preset for train/dev/test splits '''
    filename_train    = 'Restaurants_Train14.xml'
    filename_test     = 'Restaurants_Test14.xml'
    filename_preset   = 'RESTAURANTS14.txt'
    presetDataset     =  True
    semevalyear       = '2014'
    rootFolder        =  os.environ['USERPROFILE']
    model_env        =   rootFolder + '/downloads/wose-main/source-code/wose_model'
    pathToDatasets    =  '/downloads/wose-main/datasets/'
    dataset           =  str(filename_preset[0:len(filename_preset)-4]).lower()
    extract_preds     = True # extract golden and predicted aspects

    ''' set training parameters '''
    train_embeddings  =  True # train embeddings
    train_postags_emb =  False # train pos-tag embeddings
    elmo_embs         =  True # set True for ELMo embeddings or False for pre-trained fasttext
    pre_trained_embs  =  True # True : use pre-trained / False: random initialized
    postag_one_hot    =  True # True: use one-hot vector / False: random initialized
    nepochs           =  100 # number of epochs to train
    dropout           =  0.5 # dropout
    batch_size        =  32 if elmo_embs else 32
    lr_method         =  'adam'
    pre_trained_emb   =  'fasttext'
    lr                =  3e-3 if elmo_embs else 1.25e-3 # learning rate for optimizer
    l2_regul          =  1e-3 # for weight regularization
    l1_regul          =  6e-3 # for weight regularization
    rel_pos_emb       =  True # select False for Absolute Position encoding, true for Relative aware position

    overfit_threshold =  5e3
    use_sent_level   =   True # True for Sentence Level Encoder
    use_word_level   =   True # True for Word Level Encoder
    use_pos_tags      =  False # True for Part of Speech Tags
    n_heads           =  1 # for attention heads
    n_stacks          =  1 # for attention layers
    ntags             =  3
    vocab_tags        =  {'O': 0, 'B-A': 1, 'I-A': 2} # BIO tags scheme
    # embeddings
    dim_word          =  1024 if elmo_embs else 300
    use_postag_len    =  True # for pos-tags embedding size
    dim_postag        =  46 # dimension of one-hot vector for pos-tags indices
    kfold_num         =  5
    use_crf           =  True # True crf / False soft-max
    use_posrules      =  False # True for a simplified version of pos-tags
    display_step      =  50
    hidden_size       =  1024 if elmo_embs else 300  # blstm on postags, word embeddings
    num_layers        =  1 # for stack blstm


