import tensorflow as tf # tf 1.15.4
_config = tf.compat.v1.ConfigProto()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
_config.gpu_options.allow_growth = False
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

''' set model environment '''
os.chdir(os.environ['USERPROFILE'] + '/downloads/wose-master/source-code')
# import local libraries
import data_utils as utils
import pre_trained_w2fvec as w2fvec
from learn_metrics import calcMetric
from wose_model import woseModel
from configs import Config
# load model parameters
config = Config()

''' load dataset '''
sentences_seqs_pad, postags_seqs_pad,sentences_lengths,_vocab_len,_maxlen,_vocab_postags,labels_seqs_pad,index2word_map,_postags_embs,sentences_seqs,golden_aspects = utils.loadSemevalR()

''' get pretrained embeddings / zero pad is included '''
if config.pre_trained_embs and not config.elmo_embs:
    if config.pre_trained_emb == 'fasttext':
        print('Fasttext pre-trained embeddings')
        print('-------------------------------')
        embeddings = w2fvec.getPretrainedWordVextors(index2word_map)

''' data pipeline '''
x_ = np.asarray(sentences_seqs) if config.elmo_embs else np.array(sentences_seqs_pad,dtype= np.int32)
x_tokens = np.asarray(sentences_seqs)
x_postags = np.array(postags_seqs_pad,dtype=np.int32)
seqlengths = np.array(sentences_lengths,dtype=np.int32)
y_ = np.array(labels_seqs_pad,dtype=np.int32)

# monitor test accuracy for every experiment
aspects_scores = []
f1_list = []
aspects_score12 = []
aspects_score012 = []
os.chdir(os.environ['USERPROFILE'])

'''' prepare cross validation '''
kfold = KFold(config.kfold_num,shuffle= True)
if config.presetDataset:
    ids_train,ids_dev,ids_test = utils.read_preset_dataset_idxs()
    # test dataset
    print('collecting test dataset...')
    x_test = x_[-ids_test:]
    x_test_tok = x_tokens[-ids_test:]
    x_test_golden_aspects = golden_aspects[-ids_test:]
    y_test = y_[-ids_test:]
    seqlen_test=seqlengths[-ids_test:]
    x_postags_test = x_postags[-ids_test:]
    # train dataset for k-fold cross validation
    x_t = x_[:(ids_train+ids_dev)]
    #x_t_tok = x_tokens[:(ids_train+ids_dev)]
    y_t = y_[:(ids_train+ids_dev)]
    seqlengths_t = seqlengths[:(ids_train+ids_dev)]
    x_postags_t = x_postags[:(ids_train+ids_dev)]

# store predictions an cm data
predictions_list= []
acc_list = []

# enumerate splits
for train_idx,dev_idx in kfold.split(x_t):
    print('creating train/dev datasets...')
    train_idx = np.concatenate([train_idx,dev_idx[:dev_idx.__len__()//2]])
    dev_idx = dev_idx[-dev_idx.__len__()//2:]
    # train dataset
    x_train =x_t[train_idx]
    y_train=y_t[train_idx]
    seqlen_train = seqlengths_t[train_idx]
    x_postags_train= x_postags_t[train_idx]
    # dev dataset
    x_dev = x_[dev_idx]
    y_dev =y_[dev_idx]
    seqlen_dev = seqlengths[dev_idx]
    x_postags_dev =x_postags[dev_idx]

    print('dat: ' + str(len(x_))  + ' train/dev/test ' + str(len(x_train)) + '/' +str(len(x_dev)) +'/' + str(len(x_test)))

    # calculate training iterations
    training_iters = int(config.nepochs*(int(len(x_train))/config.batch_size))

    # the number of test data to evaluate periodically
    test_eval_batch = config.batch_size if config.elmo_embs else 800

    print()
    print('Training File: ' + str(config.filename_preset[:len(config.filename_preset)-4]))
    print('Model Parameters')
    print('-------------------')
    print('max sentence sequence: ' + str(_maxlen))
    print('n_hidden: ' + str(config.hidden_size))
    print('learning rate:' + str(config.lr))
    print('embedding_size: ' + str(config.dim_word))
    print('train word embeddings: ' + str(config.train_embeddings))
    print('postag embedding_size: ' + str(config.dim_postag))
    print('postag one hot: ' + str(config.postag_one_hot))
    print('train postag embeddings: ' + str(config.train_postags_emb))
    print('base_dropout: ' + str(config.dropout))
    print('num_heads: ' + str(config.n_heads))
    print('num_stacks: ' + str(config.n_stacks))
    print('num_blstm_layers: ' + str(config.num_layers))
    print('n_epochs: ' + str(config.nepochs))
    print('l1 regul: ' + str(config.l1_regul))
    print('l2 regul: ' + str(config.l2_regul))
    print('learning method: ' + str(config.lr_method))
    print('avg batch_size: ' + str(config.batch_size))
    print('use crf: ' + str(config.use_crf))
    print('use pos-tag rules: ' + str(config.use_posrules))
    print('sentence_level_only: ' + str(config.use_sent_level))
    print('use_word_level: ' + str(config.use_word_level))
    print('use pos-tags: ' + str(config.use_pos_tags))
    print('use rel_pos_emb: ' + str(config.rel_pos_emb))
    print('use elmo_embs: ' + str(config.elmo_embs))
    print('-------------------')
    print()
    print('training iterations: ' + str(training_iters))
    print('training the Word and Sentence level model...')
    step = 1
    epoch = 1
    graph = tf.Graph()
    with graph.as_default():
        model = woseModel( _vocab_len=_vocab_len, _maxlen=_maxlen,_vocab_postags=_vocab_postags)
        sess = model.initialize_session()
        with sess.as_default():
            # build aspect model
            model.build()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                decayed_lr = tf.train.exponential_decay(config.lr, step, training_iters,0.95, staircase=True)
                if config.lr_method =='rmsprop': # rms optimizer
                    _optimizer = tf.train.RMSPropOptimizer(decayed_lr, epsilon=1e-6)
                elif config.lr_method =='adam': # adam optimizer
                    _optimizer = tf.train.AdamOptimizer(decayed_lr, epsilon=1e-6)
                elif config.lr_method =='sgd':
                    _optimizer = tf.train.GradientDescentOptimizer(decayed_lr/config.batch_size)
                else:
                    print('Setup the optimization method!')

            with tf.name_scope('apply_gradient_norm'):
                gvs = _optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1, 1.), var) for grad, var in gvs]
                optimizer = _optimizer.apply_gradients(capped_gvs)

            # run and train the model
            sess.run(tf.global_variables_initializer())

            if config.pre_trained_embs and not config.elmo_embs:
                sess.run(model.embedding_init,feed_dict={model.embedding_placeholder: embeddings})
            if config.postag_one_hot and config.use_sent_level and config.use_pos_tags:
                sess.run(model.postag_embedding_init,feed_dict={model.postag_embedding_placeholder: _postags_embs})

            print('Training Parameters:',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/train', graph=tf.get_default_graph())
            dev_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/dev',graph=tf.get_default_graph())
            # initiate train/test accuracies
            acc = 0
            asf = [0,0,0]
            _acc = 0
            _count = 0

            var_dropout = utils.varDropout()

            # keep training until reach max iterations
            while step <= training_iters:
                # get train batch
                batch_x, batch_y, batch_lengths,batch_postags =  utils.next_batch(utils.varBatch(), x_train,y_train, seqlen_train, True,x_postags_train)

                # monitor training accuracy information
                summary,_ = sess.run([merged,optimizer],
                        feed_dict={model.word_ids: batch_x,
                                    model.postag_ids:batch_postags,
                                    model.labels: batch_y,
                                    model.seqlens:batch_lengths,
                                    model.dropout: var_dropout,
                                    model.is_training:True})

                # add to summaries
                train_writer.add_summary(summary, step)

                # run optimization op (backprop)
                sess.run(optimizer, feed_dict={model.word_ids: batch_x,
                                    model.postag_ids: batch_postags,
                                    model.labels: batch_y,
                                    model.seqlens: batch_lengths,
                                    model.dropout: var_dropout,
                                    model.is_training: True})

                if step % (training_iters//config.nepochs) == 0:
                    # evaluate current epoch
                    print("-"*60)
                    print("Current Test Results, Epoch:", str(epoch) + "/" + str(config.nepochs))
                    print("-"*60)
                    # calculate samples to evaluate in test dataset
                    test_len = int(x_test.shape[0])
                    # init partial lists metrices
                    list_partial_acc = []
                    list_aspect_score = []
                    list_aspect_score_macro = []
                    list_partial_cm = []
                    partial_eval = False
                    list_aspect_score_macrof0 = []

                    if test_len > test_eval_batch:
                        tmp_predictions_list = []
                        # mark the index of the data
                        eval_idx_start = 0
                        eval_idx_end = test_eval_batch #- 1
                        # mark partial evaluation
                        partial_eval = True
                        # partition the test data
                        if test_len % test_eval_batch == 0:
                            eval_range = (test_len//test_eval_batch)
                        else :
                            eval_range = (test_len//test_eval_batch)  + 1
                        for _ in range(eval_range):
                            tmp_test_data = x_test[eval_idx_start:eval_idx_end]
                            tmp_test_label = y_test[eval_idx_start:eval_idx_end]
                            tmp_test_seqs = seqlen_test[eval_idx_start:eval_idx_end]
                            tmp_x_postags_test =x_postags_test[eval_idx_start:eval_idx_end]

                            if config.use_crf:
                                # get tag scores and transition params of CRF
                                viterbi_sequences = []
                                logits, trans_params = sess.run([model.logits, model.trans_params],
                                    feed_dict={model.word_ids: tmp_test_data,
                                            model.dropout: 0.0,
                                            model.seqlens: tmp_test_seqs,
                                            model.postag_ids: tmp_x_postags_test,
                                            model.is_training:False})

                                # iterate over the sentences because no batching in vitervi_decode
                                for logit, sequence_length in zip(logits, tmp_test_seqs):
                                    logit = logit[:sequence_length] # keep only the valid steps
                                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                            logit, trans_params)
                                    viterbi_sequences += [viterbi_seq]
                                tmp_test_label_pred = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)

                            else:
                                # calculate batch accuracy and print
                                tmp_test_label_pred = sess.run(model.labels_pred,
                                    feed_dict={model.word_ids: tmp_test_data,
                                            model.dropout: 0.0,
                                            model.seqlens: tmp_test_seqs,
                                            model.postag_ids:tmp_x_postags_test,
                                            model.is_training:False})

                            partial_acc , _asf, _asf12, _asf012,acc_l = utils.run_evaluate(tmp_test_label,tmp_test_label_pred,tmp_test_seqs)
                            # store partial predictions
                            tmp_predictions_list.extend(tmp_test_label_pred)
                            # store partial accuracy
                            list_partial_acc.append(partial_acc)
                            list_aspect_score.append(_asf)
                            list_aspect_score_macro.append(np.array(list(_asf12),dtype=np.float32))
                            list_aspect_score_macrof0.append(np.array(list(_asf012),dtype=np.float32))

                            cm = tf.confusion_matrix(np.reshape(tmp_test_label,[-1]),np.reshape(tmp_test_label_pred,[-1]),num_classes=_maxlen)
                            # get confusion matrix values / store partial confusion matrix
                            list_partial_cm.append(sess.run(cm))

                            # feed test evaluation with new values
                            eval_idx_start+= test_eval_batch
                            eval_idx_end += test_eval_batch
                            if eval_idx_end > test_len:
                                eval_idx_end = test_len

                    else :
                        # evaluate all test partitions
                        test_data = x_test[:test_len]
                        test_label = y_test[:test_len]
                        test_seqs = seqlen_test[:test_len]

                        # calculate overall accuracy
                        overall_acc = sess.run(model.accuracy,
                                    feed_dict={model.word_ids: test_data,
                                            model.labels: test_label,
                                            model.seqlens: test_seqs,
                                            model.dropout: 0.0,
                                            model.postag_ids:x_postags_test,
                                            model.is_training:False})
                        if config.use_crf:
                            # get tag scores and transition params of CRF
                            viterbi_sequences = []
                            logits, trans_params = sess.run([model.logits, model.trans_params],
                                    feed_dict={model.word_ids: test_data,
                                            model.dropout: 0.0,
                                            model.seqlens: test_seqs,
                                            model.postag_ids:x_postags_test,
                                            model.is_training:False})

                            # iterate over the sentences because no batching in vitervi_decode
                            for logit, sequence_length in zip(logits, test_seqs):
                                logit = logit[:sequence_length] # keep only the valid steps
                                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                        logit, trans_params)
                                viterbi_sequences += [viterbi_seq]
                            predicted = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)
                            #store predictions
                            predictions_list.append(predicted)

                        else:
                            # calculate batch accuracy and print
                            predicted = sess.run(model.labels_pred,
                                    feed_dict={model.word_ids: test_data,
                                            model.dropout: 0.0,
                                            model.seqlens: test_seqs,
                                            model.postag_ids:x_postags_test,
                                            model.is_training:False})

                            #store predictions
                            predictions_list.append(predicted)

                        accuracy , _asf, _asf12,_asf012,acc_l = utils.run_evaluate(test_label,predicted,test_seqs)

                        cm = tf.confusion_matrix(np.reshape(test_label,[-1]),np.reshape(predicted,[-1]),10)
                        # get confusion matrix values
                        tf_cm = sess.run(cm)

                    if partial_eval:
                    #store predictions
                        predictions_list.append(tmp_predictions_list)
                        accuracy = np.ma.average(list_partial_acc)
                        _asf = np.average(list_aspect_score,axis=0)

                        _asf12 =np.average(list_aspect_score_macro,axis=0)
                        _asf012 = np.average(list_aspect_score_macrof0,axis=0)
                        tf_cm = np.ma.sum(list_partial_cm,axis=0)

                    print("\nOverall Testing Accuracy: ", "{:.5f}".format(accuracy), '\naspects score:',_asf, '\naspects score macro12:',_asf12, '\naspects score macro 012:',_asf012)
                    # monitor f1 score
                    f1_list.append(_asf[-1])
                    epoch+=1
                    print("-"*60)

                # monitor train accuracy information in python window
                if step % config.display_step ==0:

                    if config.use_crf:
                        # get tag scores and transition params of CRF
                        viterbi_sequences = []
                        logits, trans_params = sess.run([model.logits, model.trans_params],
                            feed_dict={model.word_ids: batch_x,
                                        model.postag_ids: batch_postags,
                                        model.labels: batch_y,
                                        model.seqlens: batch_lengths,
                                        model.dropout: var_dropout,
                                        model.is_training: True})

                        # iterate over the sentences because no batching in vitervi_decode
                        for logit, sequence_length in zip(logits, batch_lengths):
                            logit = logit[:sequence_length] # keep only the valid steps
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    logit, trans_params)
                            viterbi_sequences += [viterbi_seq]
                        #convert for score calculation
                        labels_pred = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)

                    else: #softmax
                        # calculate batch accuracy
                        labels_pred = sess.run(model.labels_pred,
                            feed_dict={model.word_ids: batch_x,
                                        model.postag_ids: batch_postags,
                                        model.labels: batch_y,
                                        model.seqlens: batch_lengths,
                                        model.dropout: var_dropout,
                                        model.is_training: True})

                    acc , asf, asf12, asf012,acc_l =  \
                    utils.run_evaluate(batch_y,labels_pred,batch_lengths)

                    # calculate batch loss
                    loss = sess.run(model.loss,
                            feed_dict={model.word_ids: batch_x,
                                    model.postag_ids:batch_postags,
                                    model.labels:batch_y,
                                    model.seqlens:batch_lengths,
                                    model.dropout: var_dropout,
                                    model.is_training:True})

                # monitor dev accuracy information
                if step % 5 == 0:
                    # get dev batch
                    batch_x, batch_y, batch_lengths, batch_postags = utils.next_batch((utils.varBatch(),int(x_dev.shape[0]))[utils.varBatch() > int(x_dev.shape[0])], x_dev, y_dev, seqlen_dev, True,x_postags_dev)

                    # calculate batch loss
                    loss = sess.run(model.loss,
                            feed_dict={model.word_ids: batch_x,
                                    model.postag_ids:batch_postags,
                                    model.labels:batch_y,
                                    model.seqlens:batch_lengths,
                                    model.dropout: var_dropout,
                                    model.is_training:False})

                    summary, _acc = sess.run([merged,model.accuracy],
                                    feed_dict={model.word_ids: batch_x,
                                    model.postag_ids:batch_postags,
                                    model.labels: batch_y,
                                    model.seqlens:batch_lengths,
                                    model.dropout:var_dropout,
                                    model.is_training:False})

                    if config.use_crf:
                        # get tag scores and transition params of CRF
                        viterbi_sequences = []
                        logits, trans_params = sess.run([model.logits, model.trans_params],
                            feed_dict={model.word_ids: batch_x,
                                        model.postag_ids: batch_postags,
                                        model.labels: batch_y,
                                        model.seqlens: batch_lengths,
                                        model.dropout: var_dropout,
                                        model.is_training: False})

                        # iterate over the sentences because no batching in vitervi_decode
                        for logit, sequence_length in zip(logits, batch_lengths):
                            logit = logit[:sequence_length] # keep only the valid steps
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    logit, trans_params)
                            viterbi_sequences += [viterbi_seq]
                        #convert for score calculation
                        labels_pred = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)

                    else:
                        # calculate batch accuracy
                        labels_pred = sess.run(model.labels_pred,
                            feed_dict={model.word_ids: batch_x,
                                        model.postag_ids: batch_postags,
                                        model.labels: batch_y,
                                        model.seqlens: batch_lengths,
                                        model.dropout: var_dropout,
                                        model.is_training: False})

                    _acc , _asf, _asf12, _asf012,acc_l =  \
                    utils.run_evaluate(batch_y,labels_pred,batch_lengths)

                    # calculate new dropout
                    _count = calcMetric.calcOverfit(asf[-1],_asf[-1],_count)

                    # set a value to prevent overfit early-stop
                    if _count > config.overfit_threshold:
                        print('Overfit Identidied')
                        step = training_iters

                    dev_writer.add_summary(summary, step )

                step += 1

            print("Optimization Finished!")
            # calculate samples to evaluate in test dataset
            test_len = int(x_test.shape[0])
            # init partial lists metrices
            list_partial_acc = []
            list_aspect_score = []
            list_aspect_score_macro = []
            list_partial_cm = []
            partial_eval = False
            list_aspect_score_macrof0 = []

            if test_len > test_eval_batch:
                # mark the index of the data
                eval_idx_start = 0
                eval_idx_end = test_eval_batch #- 1
                # mark partial evaluation
                partial_eval = True
                # partition the test data
                if test_len % test_eval_batch == 0:
                    eval_range = (test_len//test_eval_batch) # + 1
                else :
                    eval_range = (test_len//test_eval_batch)  + 1
                for _ in range(eval_range):
                    tmp_test_data = x_test[eval_idx_start:eval_idx_end]
                    tmp_test_label = y_test[eval_idx_start:eval_idx_end]
                    tmp_test_seqs = seqlen_test[eval_idx_start:eval_idx_end]
                    tmp_x_postags_test =x_postags_test[eval_idx_start:eval_idx_end]

                    if config.use_crf:
                        # get tag scores and transition params of CRF
                        viterbi_sequences = []
                        logits, trans_params = sess.run([model.logits, model.trans_params],
                            feed_dict={model.word_ids: tmp_test_data,
                                    model.dropout: 0.0,
                                    model.seqlens: tmp_test_seqs,
                                    model.postag_ids: tmp_x_postags_test,
                                    model.is_training:False})

                        # iterate over the sentences because no batching in vitervi_decode
                        for logit, sequence_length in zip(logits, tmp_test_seqs):
                            logit = logit[:sequence_length] # keep only the valid steps
                            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    logit, trans_params)
                            viterbi_sequences += [viterbi_seq]
                        tmp_test_label_pred = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)

                    else:
                        # calculate batch accuracy and print
                        tmp_test_label_pred = sess.run(model.labels_pred,
                            feed_dict={model.word_ids: tmp_test_data,
                                    model.dropout: 0.0,
                                    model.seqlens: tmp_test_seqs,
                                    model.postag_ids:tmp_x_postags_test,
                                    model.is_training:False})

                    partial_acc , _asf, _asf12, _asf012,acc_l = utils.run_evaluate(tmp_test_label,tmp_test_label_pred,tmp_test_seqs)

                    # store partial accuracy
                    list_partial_acc.append(partial_acc)
                    list_aspect_score.append(_asf)
                    list_aspect_score_macro.append(np.array(list(_asf12),dtype=np.float32))
                    list_aspect_score_macrof0.append(np.array(list(_asf012),dtype=np.float32))

                    cm = tf.confusion_matrix(np.reshape(tmp_test_label,[-1]),np.reshape(tmp_test_label_pred,[-1]),num_classes=_maxlen)
                    # get confusion matrix values / store partial confusion matrix
                    list_partial_cm.append(sess.run(cm))

                    # feed test evaluation with new values
                    eval_idx_start+= test_eval_batch
                    eval_idx_end += test_eval_batch
                    if eval_idx_end > test_len:
                        eval_idx_end = test_len

            else :
                # evaluate all test partitions
                test_data = x_test[:test_len]
                test_label = y_test[:test_len]
                test_seqs = seqlen_test[:test_len]

                # calculate overall accuracy
                overall_acc = sess.run(model.accuracy,
                            feed_dict={model.word_ids: test_data,
                                    model.labels: test_label,
                                    model.seqlens: test_seqs,
                                    model.dropout: 0.0,
                                    model.postag_ids:x_postags_test,
                                    model.is_training:False})
                if config.use_crf:
                    # get tag scores and transition params of CRF
                    viterbi_sequences = []
                    logits, trans_params = sess.run([model.logits, model.trans_params],
                            feed_dict={model.word_ids: test_data,
                                    model.dropout: 0.0,
                                    model.seqlens: test_seqs,
                                    model.postag_ids:x_postags_test,
                                    model.is_training:False})

                    # iterate over the sentences because no batching in vitervi_decode
                    for logit, sequence_length in zip(logits, test_seqs):
                        logit = logit[:sequence_length] # keep only the valid steps
                        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                                logit, trans_params)
                        viterbi_sequences += [viterbi_seq]
                    predicted = utils.pad_viterbi_preds(viterbi_sequences,_maxlen)

                else:
                    # calculate batch accuracy and print
                    predicted = sess.run(model.labels_pred,
                            feed_dict={model.word_ids: test_data,
                                    model.dropout: 0.0,
                                    model.seqlens: test_seqs,
                                    model.postag_ids:x_postags_test,
                                    model.is_training:False})
                    #store predictions
                    predictions_list.append(predicted)

                accuracy , _asf, _asf12,_asf012, acc_l = utils.run_evaluate(test_label,predicted,test_seqs)

                cm = tf.confusion_matrix(np.reshape(test_label,[-1]),np.reshape(predicted,[-1]),10)
                # get confusion matrix values
                tf_cm = sess.run(cm)

            if partial_eval:
                 accuracy = np.ma.average(list_partial_acc)
                 _asf = np.average(list_aspect_score,axis=0)
                 _asf12 =np.average(list_aspect_score_macro,axis=0)
                 _asf012 = np.average(list_aspect_score_macrof0,axis=0)
                 tf_cm = np.ma.sum(list_partial_cm,axis=0)

            print("\nOverall Testing Accuracy: ", "{:.5f}".format(accuracy), '\naspects score:',_asf, '\naspects score macro12:',_asf12, '\naspects score macro 012:',_asf012)
            acc_list.append(acc_l)
            aspects_scores.append(list(_asf))
            aspects_score12.append(list(np.array(list(_asf12),dtype=np.float32)))
            aspects_score012.append(list(np.array(list(_asf012),dtype=np.float32)))

    print("")
    sess.close()
    print('deleting model')
    del model

# tensorboard --logdir=/tmp/tensorflowlogs
print("f1 statistics")
print(aspects_scores)
print("f1 average")
print(np.average(aspects_scores,axis=0))
print("macro statistics labels 12")
print(aspects_score12)
print("macro average f12")
print(np.average(aspects_score12,axis=0))
print("macro statistics labels 012")
print(aspects_score012)
print("macro average f0")
print(np.average(aspects_score012,axis=0))

if config.extract_preds:
    os.chdir(os.environ['USERPROFILE'] + '/downloads/wose-master/output-data')
    arr = np.array(f1_list)
    best_pred = np.where(arr == np.amax(arr))[0][0]
    print("Best F1 score at index:",best_pred, " with value:","{:.5f}".format(np.amax(arr)))

    with open(config.dataset +'_golden_aspects.txt', 'w', encoding='utf-8') as f:
        for item in x_test_golden_aspects:
            f.write("%s\n" %item )
    f.close()

    pred_aspects = []
    for i, pred in enumerate(predictions_list[best_pred]):
        lab_pred = pred[:seqlen_test[i]].tolist()
        lab_pred_chunks = set(utils.get_chunks(lab_pred,config.vocab_tags,message = "prediction"))
        pred_chunks  =  ([[x_test_tok[i][c] for c in  range(j[1],j[2])] for j in sorted(lab_pred_chunks)])
        pred_chunks_final = []
        for chunk in pred_chunks:
            temp = ""
            for item in chunk:
                temp+= ' ' + item
            pred_chunks_final.append(temp.strip())
        pred_aspects.append(pred_chunks_final)

    with open(config.dataset + '_predicted_aspects.txt', 'w',encoding='utf-8') as f:
        for item in pred_aspects:
            f.write("%s\n" %item )
    f.close()



