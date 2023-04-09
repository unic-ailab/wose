import numpy as np
import os
import random
import re
from configs import Config
import pickle
import copy
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from libraries.baselines import Corpus
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')
# load model parameters
config =Config()

''''   '''
bad_chars = ["*", "=", "+"]
vals = np.array([config.dropout, config.dropout, config.dropout, config.dropout, config.dropout],dtype=np.float32)
vals_batch = np.array([config.batch_size, config.batch_size, config.batch_size],dtype=np.int16)
s_random = random.SystemRandom()
dataset_len={'restaurants14':79,'restaurants15':68,'restaurants16':78,'laptop14':83}

def _pad_sequences(sequences, pad_tok, max_length):

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):

    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)
    if nlevels ==1:
        return sequence_padded, sequence_length, max_length
    else:
        return sequence_padded, sequence_length

def get_chunk_type(tok, idx_to_tag):

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags,message=None):

    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == 'B':
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    #print(message + "{}{}{}".format(seq, tags, chunks))

    return chunks

def run_evaluate(labels, labels_pred, sequence_lengths):

    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    gold = []
    pred = []
    acc_list = []


    for lab, lab_pred, length in zip(labels, labels_pred,
                                        sequence_lengths):
        lab      = lab[:length].tolist()
        lab_pred = lab_pred[:length].tolist()
        acc_list.append(lab==lab_pred)
        gold+=lab
        pred+=lab_pred
        accs    += [a==b for (a, b) in zip(lab, lab_pred)]
        lab_chunks      = set(get_chunks(lab, config.vocab_tags,message = "gold standard"))
        #tmp_lab_chunks.append(lab_chunks)
        lab_pred_chunks = set(get_chunks(lab_pred,config.vocab_tags,message = "prediction"))
        #tmp_lab_pred_chunks.append(lab_pred_chunks)
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    acc = np.mean(accs)

    #include 1,2
    score = precision_recall_fscore_support(gold, pred, labels =[1,2], average='macro')
    #include 0,1,2
    score_0 = precision_recall_fscore_support(gold, pred, average='macro')

    return acc, np.round([p,r,f1],5), ["{0:.5f}".format(x) for x in score[:3]], ["{0:.5f}".format(x) for x in score_0[:3]], acc_list


def process_semeval_2014(train_filename, test_filename):
    os.chdir(config.rootFolder + config.pathToDatasets + config.dataset)
    # the train set is composed by train and trial dataset
    corpora = dict()
    corpora['dataset'] = dict()
    corpus = Corpus(ET.parse(train_filename).getroot().findall('sentence') + ET.parse(test_filename).getroot().findall('sentence'))
    corpora['dataset']['set'] = dict()
    corpora['dataset']['set']['corpus'] = corpus
    return corpora

def load_semeval_2015(filepath):
    reviews = []
    with open(filepath, encoding='utf-8') as f:
        soup = BeautifulSoup(f, "xml")
        review_tags = soup.find_all("Review")
        for j, r_tag in enumerate(review_tags):
            review = Review()
            review.id = r_tag["rid"]
            sentence_tags = r_tag.find_all("sentence")
            for s_tag in sentence_tags:
                sentence = Sentence()
                sentence.review_id = review.id
                sentence.id = s_tag["id"]
                sentence.text = s_tag.find("text").get_text()
                opinion_tags = s_tag.find_all("Opinion")
                for o_tag in opinion_tags:
                    opinion = Opinion()

                    # category
                    try:
                        opinion.category = o_tag["category"]
                    except KeyError:
                        opinion.category = None

                    # entity + attribute
                    if opinion.category and "#" in opinion.category:
                        opinion.entity, opinion.attribute = opinion.category.split("#")
                    else:
                        opinion.entity = None
                        opinion.attribute = None

                    # polarity
                    try:
                        opinion.polarity = o_tag["polarity"]
                    except KeyError:
                        opinion.polarity = None

                    try:
                        opinion.target = o_tag["target"]
                        if opinion.target == "NULL":
                            opinion.target = None
                        else:
                            opinion.start = int(o_tag["from"])
                            opinion.end = int(o_tag["to"])
                    except KeyError:
                        pass
                    sentence.opinions.append(opinion)
                review.sentences.append(sentence)
            reviews.append(review)
    return reviews

def process_semeval_2015(train_filename, test_filename):
    os.chdir(config.rootFolder + config.pathToDatasets + config.dataset)
    sentences = [] ; aspects = [] ; aspect_categories = []
    train = load_semeval_2015(train_filename)

    def checkNull(t):
        if t==None:
            return ''
        else :
            return t

    for _r in train:
        for _s in _r.sentences:
            sentences.append(clear_data(_s.text))
            aspects.append([clear_data(checkNull(_op.target)) for _op in _s.opinions])
            aspect_categories.append([clear_data(_op.category) for _op in _s.opinions])

    test = load_semeval_2015(test_filename)
    for _r in test:
        for _s in _r.sentences:
            sentences.append(clear_data(_s.text))
            aspects.append([clear_data(checkNull(_op.target)) for _op in _s.opinions])
            aspect_categories.append([clear_data(_op.category) for _op in _s.opinions])

    return sentences, aspects, aspect_categories


def extractPreprocessLog(filename):
    file = open(filename,"w")
    for i in range(len(sentences_tok)):
        file.write(str(sentences_tok[i]) + ' length: ' +  str(len(sentences_tok[i])))
        file.write('\n')
        file.write(str( str(sortAspectTerms(sentences_tok[i],aspects_tok[i])) + " length:" + str(len(sortAspectTerms(sentences_tok[i],aspects_tok[i])))))
        file.write('\n')
        file.write(str(labels_tok[i]) + ' length: ' + str(len(labels_tok[i])))
        file.write('\n\n')
    file.close()


def next_batch(num, data, labels,seqlens,_has_seqns,postags):

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    seqlens_shufle = [seqlens[ i] for i in idx]
    postags_shufle = [postags[ i] for i in idx]
    if _has_seqns is True:
        return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(seqlens_shufle),np.asanyarray(postags_shufle)
    else :
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def read_preset_dataset_idxs():
    _filename = config.rootFolder + config.pathToDatasets + config.dataset + '/' + config.filename_preset
    load_file = open(_filename, "r")
    idxs =[]
    for line in load_file:
        idxs.append(line.rstrip('\n').replace('"', ''))
    load_file.close()
    idxs =list(map(int,idxs))
    return idxs

def varDropout():
    return np.round(s_random.choice(vals),2) #keep dropout

def varBatch():
    return np.round(s_random.choice(vals_batch),2)

def pad_viterbi_preds(preds, _maxlen):
    padded_preds = []
    # loop through opinions
    for i in range(len(preds)):
        pred = preds[i]
        num_padding = _maxlen - len(pred)
        tmp_array = np.concatenate([np.array(pred),np.zeros(num_padding)])
        padded_preds.append(tmp_array)

    return padded_preds

def removeBadChars(tokens):
    _processed_tokens = []
    for t in tokens:
        if t.__len__()>1:
            _tmp = ''.join(i for i in t if not i in bad_chars)
            if _tmp.__len__()==0:
                _processed_tokens.append(t)
            else:
                _processed_tokens.append(_tmp)
        else:
            _processed_tokens.append(t)
    return _processed_tokens

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def loadSemevalR():

    def preprocessRawTokens(tokens):
        _preptoken = []
        for t in tokens:
            _t = t.lower()
            if _t.isdigit():
                _t = 'digit'
            _preptoken.append(_t)
        return _preptoken

    def build_vocab(data_dir, plain = []):
        i=0
        for fn in os.listdir(data_dir):
            if fn.endswith('.xml'):
                with open(data_dir+fn,encoding='utf-8') as f:
                    dom=ET.parse(f)
                    root=dom.getroot()
                    for sent in root.iter("sentence"):
                        text = sent.find('text').text
                        token = removeBadChars(preprocessRawTokens(word_tokenize(text)))
                        plain = plain + token
                        i+=1
            #print(i)
        vocab = sorted(set(plain))
        word_idx = {}
        for idx, word in enumerate(vocab):
            word_idx[word] = idx+1
        return word_idx, i

    def vocabPostag():
        return ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS','NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP','SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',',','.',':','$','#',"``","''",'(',')']

    def parseData(fl,_sentlens,_maxlen):
        pos_tag_list = vocabPostag()

        if config.use_posrules:
            pos_tag_list_rules = ['CC','NN','JJ','VB','RB','IN']

        tag_to_num = {tag:i+1 for i, tag in enumerate(sorted(pos_tag_list))}
        corpus = []
        corpus_tag = []
        data_X = np.zeros((_sentlens, _maxlen), np.int16)
        data_X_tag = np.zeros((_sentlens, _maxlen), np.int16)
        data_y = np.zeros((_sentlens, _maxlen), np.int16)
        data_seqlen = np.zeros(_sentlens,np.int16)
        golden_aspects = np.empty(_sentlens, dtype=list)

        with open(fl,encoding='utf-8') as f:
            dom=ET.parse(f)
            root=dom.getroot()
            data_tokens = [] #store sentences tokenized
            # iterate the review sentence
            for sx, sent in enumerate(root.iter("sentence") ) :
                if sx%10==0:
                    print('finish sentence:', str(sx))

                text = sent.find('text').text

                token = removeBadChars(preprocessRawTokens(word_tokenize(text)))
                data_tokens.append(token)
                data_seqlen[sx] = len(token)
                corpus.append(token)
                pos_tag_stf = [tag_to_num[tag] for (_,tag) in nltk.pos_tag(token)]

                ''' if pos-tag hand rules will applied '''
                if config.use_posrules:
                    pos_tag_stf_tag = [tag for (_,tag) in nltk.pos_tag(token)]
                    _pos_tag_stf = []
                    for _idx,_tag in enumerate(pos_tag_stf_tag):
                        if _tag in pos_tag_list_rules:
                            _pos_tag_stf.append(pos_tag_stf[_idx])
                        else :
                            _pos_tag_stf.append(0)
                    pos_tag_stf = _pos_tag_stf

                # write word index and tag in data_X and data_X_tag
                for wx, word in enumerate(token):
                    data_X[sx, wx] = word_idx[word]
                    data_X_tag[sx, wx] = pos_tag_stf[wx]

                # different path for semeval 14 and 15
                if config.semevalyear=='2015' or config.semevalyear=='2016':
                    # iterate the opinions
                    _aspects = []
                    tmp_start= []
                    tmp_end = []
                    tmp_start.append(-1)
                    tmp_end.append(-1)
                    for ox, opin in enumerate(sent.iter('Opinion')) :
                        # extract attibutes of Opinion
                        target, category, polarity, start, end = opin.attrib['target'], opin.attrib['category'], opin.attrib['polarity'], int(opin.attrib['from']), int(opin.attrib['to'])
                        if start == tmp_start[-1] and end == tmp_end[-1]: #prevent double similar aspects
                            continue
                        tmp_start.append(start)
                        tmp_end.append(end)
                        # print(start,end)

                        if target=='NULL' :
                             break
                        else:
                            # print(target)
                            _aspects.append(target.lower())

                        catag_main, catag_sub = category.split('#')
                        # find word index (instead of str index) if start,end is not (0,0)

                        if end != 0:
                            if start != 0:
                                start = len(word_tokenize(text[:start]))
                            end = len(word_tokenize(text[:end])) #-1
                            # for training only identify aspect word, but not polarity
                            data_y[sx, start] = 1
                            if end > start:
                                data_y[sx, start+1:end] = 2
                    golden_aspects[sx] = _aspects

                elif config.semevalyear=='2014':
                    for ox, opin in enumerate(sent.iter('aspectTerms')) :
                        _aspects = []
                        for _ox, _opin in enumerate(opin.iter('aspectTerm')):
                            # extract attibutes of Opinion
                            target, start, end = _opin.attrib['term'], int(_opin.attrib['from']), int(_opin.attrib['to'])
                            _aspects.append(target.lower())

                            if end != 0:
                                if start != 0:
                                    start = len(word_tokenize(text[:start]))
                                end = len(word_tokenize(text[:end])) #-1
                                # for training only identify aspect word, but not polarity
                                data_y[sx, start] = 1
                                if end > start:
                                    data_y[sx, start+1:end] = 2

                        golden_aspects[sx] = _aspects


        return data_X, data_X_tag,data_y, data_seqlen , data_tokens, golden_aspects

    vocab_postags = vocabPostag()

    # postag preprocess
    if config.postag_one_hot:
        _postags_indices = np.array([index[0]+1 for index in enumerate(vocab_postags)])
        _postags_indices = np.concatenate([np.zeros(1),_postags_indices],axis=0)
        #convert to one hot
        _postags_embs = np.eye(int(config.dim_postag ))[np.int32(_postags_indices)]
    else :
        _postags_embs = np.zeros(1)

    os.chdir(config.rootFolder + config.pathToDatasets + '/' + config.dataset)
    _maxlen = dataset_len[config.dataset]

    word_idx, _sentlens = build_vocab(os.getcwd() +'/')

    idx_2wordmap = {index:word for word, index in word_idx.items()}

    # get train/dev/test splits
    train_len,dev_len ,test_len = read_preset_dataset_idxs()
    train_len = train_len + dev_len

    train_x, train_x_tag, train_y, train_seqlen, train_x_tokens,golden_asps_train_x =parseData(config.filename_train,train_len,_maxlen)

    test_x, test_x_tag, test_y,test_seqlen, test_x_tokens,golden_asps_test_x = parseData(config.filename_test,test_len,_maxlen)


    data_X = np.concatenate([train_x,test_x])
    data_x_tag = np.concatenate([train_x_tag,test_x_tag])
    data_x_golden_aspects= np.concatenate([golden_asps_train_x,golden_asps_test_x])
    data_y = np.concatenate([train_y,test_y])
    data_seqlen = np.concatenate([train_seqlen,test_seqlen])
    data_X_tokens = np.concatenate([train_x_tokens,test_x_tokens])
    x_ut_pad_tok,_,_  = pad_sequences(data_X_tokens,'',1) # process sentenses for elmo embeddings


    return data_X, data_x_tag, data_seqlen, idx_2wordmap.__len__(), _maxlen, _postags_embs.__len__(), data_y, idx_2wordmap, _postags_embs, x_ut_pad_tok, data_x_golden_aspects





