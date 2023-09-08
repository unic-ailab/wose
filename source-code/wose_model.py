import tensorflow as tf # ver 1.15.4
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import regularizers
from configs import Config
from BModel import BModel
# load model parameters
config = Config()

class woseModel(BModel):
    def __init__(self, _vocab_len, _maxlen, _vocab_postags):
        super(woseModel, self).__init__(config)
        self._vocab_len = _vocab_len
        self._vocab_postags = _vocab_postags
        self._maxlen = _maxlen
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],name="dropout") ##placeholders
        if config.elmo_embs:
            self.word_ids = tf.placeholder(tf.string, shape=[None, self._maxlen],name="word_ids")
            self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=config.train_embeddings)
        else:
            self.word_ids = tf.placeholder(tf.int32, shape=[None, self._maxlen],name="word_ids")
        self.postag_ids = tf.placeholder(tf.int32, shape=[None, self._maxlen],name="postag_ids")
        self.seqlens = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None],name="labels")
        self.embedding_placeholder = tf.placeholder(tf.float32,[_vocab_len+1,config.dim_word])
        self.postag_embedding_placeholder = tf.placeholder(tf.float32,[self._vocab_postags,config.dim_postag])
        self.is_training = tf.placeholder(tf.bool, name="is_training")

    def ElmoEmbeddings(self,_xtoks, _slens):
        return self.elmo(inputs={"tokens": _xtoks,"sequence_len":_slens},signature="tokens",as_dict=True)["elmo"]

    def relative_positional_encoding(self,k):

        def shape_list(x):
            """Deal with dynamic shape in tensorflow cleanly."""
            static = x.shape.as_list()
            dynamic = tf.shape(x)
            return [dynamic[i] if s is None else s for i, s in enumerate(static)]
        batch, heads, sequence, features = shape_list(k)
        E = tf.get_variable('E', [heads, sequence, features])
        k_ = tf.transpose(k, [1, 0, 2, 3])
        k_ = tf.reshape(k_, [heads, batch * sequence, features])
        rel = tf.matmul(k_, E, transpose_b=True)
        rel = tf.reshape(rel, [heads, batch, sequence, sequence])
        rel = tf.pad(rel, ((0, 0), (0, 0), (0, 0), (1, 0)))
        rel = tf.reshape(rel, (heads, batch, sequence+1, sequence))
        rel = rel[:, :, 1:]
        rel = tf.transpose(rel, [1, 0, 2, 3])
        return rel


    def positional_encoding(self,inputs, num_units=config.dim_word, zero_pad=True,
                            scale=True,scope="positional_encoding",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)] for pos in range(inputs.get_shape().as_list()[1])],dtype=np.float32)
            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)

            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                        lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

            if scale:
                outputs = outputs * num_units**0.5

            return outputs

    def layer_norm(self,inputs,epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self,inputs, encoded_output, num_units=None,
                            num_heads=config.n_heads,
                            masking=False,
                            scope="multihead_attention", reuse=None, decoding=False,relative=False):

        if decoding: #sent_level_encoder, word_level_encoder
            queries, keys, values = inputs, encoded_output, encoded_output
        else:
            queries, keys, values = inputs, inputs, inputs

        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu,kernel_regularizer=regularizers.l1_l2(l1=config.l1_regul, l2=config.l2_regul),
    bias_regularizer=regularizers.l2(config.l2_regul),
    activity_regularizer=regularizers.l2(config.l2_regul))
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu,kernel_regularizer=regularizers.l1_l2(l1=config.l1_regul, l2=config.l2_regul),
    bias_regularizer=regularizers.l2(config.l2_regul),
    activity_regularizer=regularizers.l2(config.l2_regul))
            V = tf.layers.dense(values, num_units, activation=tf.nn.relu,kernel_regularizer=regularizers.l1_l2(l1=config.l1_regul, l2=config.l2_regul),
    bias_regularizer=regularizers.l2(config.l2_regul),
    activity_regularizer=regularizers.l2(config.l2_regul))

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            if relative:
                with tf.variable_scope("relpos_enc", reuse=reuse):
                    # reshape queries for use in relative attention
                    k_ = tf.reshape(K_,[-1, num_heads, K_.shape[1], K_.shape[2]])
                    rel = tf.reshape(self.relative_positional_encoding(k_),[-1, k_.shape[2], k_.shape[2]])

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

            # relative position encoding
            if relative:
                outputs += rel

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            if masking:
                key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
                key_masks = tf.tile(key_masks, [num_heads, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            # Activation
            outputs = tf.nn.softmax(outputs)

            # Query Masking
            if masking:
                query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
                query_masks = tf.tile(query_masks, [num_heads, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
                outputs *= query_masks

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=self.dropout, training=self.is_training)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 )

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.layer_norm(outputs)

        return outputs

    def feedforward(self,inputs, num_units=None,scope=None):
        with tf.variable_scope(scope):
            if num_units==None:
                num_units = [4*inputs.get_shape().as_list()[-1],inputs.get_shape().as_list()[-1]]

            hidden_layer = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu, use_bias=True)
            output_layer = tf.layers.dense(hidden_layer, num_units[1], activation=tf.nn.relu, use_bias=True)

            # residual connection
            output_layer += inputs

            # Normalize
            outputs = self.layer_norm(output_layer, scope="layer_norm")

        return outputs

    # Decoder Stacked layers  sent_level_encoder, word_level_encoder
    def decoding_stack(self,inputs, enc_input, num_stack=config.n_stacks,masking=False):
        for idx in range(num_stack):
            with tf.variable_scope("decoding_stack_{}".format(idx)):
                hidden =  self.multihead_attention(inputs=inputs, encoded_output=None, decoding=False,masking=False , scope="mask_dec")
                enc_added =  self.multihead_attention(hidden, encoded_output=enc_input, decoding=True, scope="self_att_dec",masking=masking)
                inputs =  self.feedforward(enc_added,num_units=[config.hidden_size,  config.hidden_size], scope="dec_pffn")

        return inputs

    def build(self):
        if config.elmo_embs: #elmo embeddings
            with  tf.name_scope('word_embeddings'):
                word_embeddings =  tf.layers.dropout(self.ElmoEmbeddings(self.word_ids,self.seqlens), rate=self.dropout, training=self.is_training)
                word_embeddings = tf.reshape(word_embeddings,[-1,self._maxlen,config.dim_word])

        else: # word embeddings
            # define embeddings layer
            with  tf.name_scope('word_embeddings'):
                if config.pre_trained_embs: # use pre-trained embeddings
                    _word_embeddings = tf.Variable(tf.constant(0.0, shape=[self._vocab_len+1, config.dim_word]), trainable=config.train_embeddings, name='_word_embeddings')
                    self.embedding_init = _word_embeddings.assign(self.embedding_placeholder)
                else: #random initiate embeddings
                    _word_embeddings = tf.Variable(tf.random_uniform([self._vocab_len+1 , config.dim_word], -1.0, 1.0),name='_word_embeddings')

                word_embeddings =  tf.layers.dropout(tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings"), rate=self.dropout, training=self.is_training)

        if config.use_sent_level:
            if config.use_pos_tags:
                with tf.variable_scope("postags_embeddings"):
                    # get postags embeddings matrix
                    if config.postag_one_hot:
                        _postag_embeddings = tf.Variable(tf.constant(0.0, shape=[self._vocab_postags, config.dim_postag]), trainable=config.train_postags_emb, name='_postag_embeddings')
                        self.postag_embedding_init = _postag_embeddings.assign(self.postag_embedding_placeholder)
                    else:
                        _postag_embeddings = tf.get_variable(name="_postag_embeddings",dtype=tf.float32,shape=[self._vocab_postags+1, config.dim_postag])

                    # postag embeddings
                    postag_embeddings = tf.layers.dropout(tf.nn.embedding_lookup(_postag_embeddings,self.postag_ids, name="postag_embeddings"), rate=self.dropout, training=self.is_training)

            ## sentence level encoder
            # bi-lstm on postags
            if config.use_pos_tags:
                with tf.variable_scope('blstm_postags') as scope:
                    if config.num_layers==1:
                        cell_fw_postags = tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True )
                        cell_bw_postags = tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True )
                        _output_postags, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_postags, cell_bw_postags, postag_embeddings,sequence_length=self.seqlens, dtype=tf.float32)
                        # concat output
                        output_postags = tf.layers.dropout(tf.concat(_output_postags, axis=-1), rate=self.dropout, training=self.is_training)
                    elif config.num_layers>1:# stack blstm
                        # Define LSTM cells
                        cell_fw_postags = [tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True ) for layer in range(config.num_layers)]
                        cell_bw_postags = [tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True ) for layer in range(config.num_layers)]
                        _output_postags, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cell_fw_postags, cells_bw=cell_bw_postags, inputs=postag_embeddings,sequence_length=self.seqlens, dtype=tf.float32)
                        output_postags = tf.layers.dropout(_output_postags, rate=self.dropout, training=self.is_training)

            # bi-lstm words
            with tf.variable_scope('blstm_words', initializer=tf.contrib.layers.xavier_initializer()) as scope:
                if config.num_layers ==1:
                    cell_fw_words = tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True )
                    cell_bw_words = tf.contrib.rnn.LSTMCell(config.hidden_size//2,state_is_tuple=True )
                    _output_words, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_words, cell_bw_words, word_embeddings,sequence_length=self.seqlens, dtype=tf.float32)
                    # read and concat output
                    output_words = tf.layers.dropout(tf.concat(_output_words, axis=-1), rate=self.dropout, training=self.is_training)
                elif config.num_layers>1:# stack blstm
                    # Define LSTM cells
                    cell_fw_words = [tf.contrib.rnn.LSTMCell(config.hidden_size ,state_is_tuple=True ) for layer in range(config.num_layers)]
                    cell_bw_words = [tf.contrib.rnn.LSTMCell(config.hidden_size,state_is_tuple=True ) for layer in range(config.num_layers)]
                    _output_words, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cell_fw_words, cells_bw=cell_bw_words, inputs=word_embeddings,sequence_length=self.seqlens, dtype=tf.float32)
                    output_words = tf.layers.dropout(_output_words, rate=self.dropout, training=self.is_training)

            with tf.variable_scope("sent_level_encoder"):
                if config.use_pos_tags:
                    _sent_level_encoder =  tf.math.add(output_words,output_postags) # language fusing
                else: # w/o pos-tags
                    _sent_level_encoder = output_words

                encoder_s = _sent_level_encoder
                # multi head attention memory network (MN)
                for i in range(config.n_stacks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        encoder_s = self.multihead_attention(encoder_s,encoded_output=None,decoding=False, masking=False, scope="self_att_enc")
                        encoder_s = self.feedforward(encoder_s,scope='enc_pffn')

                sent_level_encoder =  tf.math.add(encoder_s,_sent_level_encoder)

        if config.use_word_level:
            ## word level encoder
            with tf.variable_scope("word_level_encoder"):
                # encode_plus positions
                if config.rel_pos_emb: #relative positional emnbeddings
                    encoder = word_embeddings
                else : # absolute positional embeddings
                    encoder = word_embeddings + self.positional_encoding(self.word_ids) #+
                #multi head attention
                for i in range(config.n_stacks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        if config.rel_pos_emb: # use relative positional embeddinjgs
                            word_level_encoder = self.multihead_attention(encoder,encoded_output=None,decoding=False, masking=False, scope="self_att_enc",relative=True)
                        else: # dont use relative positional embeddings
                            word_level_encoder = self.multihead_attention(encoder,encoded_output=None,decoding=False, masking=False, scope="self_att_enc")

                        word_level_encoder = self.feedforward(word_level_encoder,scope='enc_pffn')

        if config.use_word_level and config.use_sent_level:
            ## disassembling layer
            with tf.variable_scope("decoder"):
                if config.use_sent_level and config.use_word_level: #WoSe
                    dec_outputs = self.decoding_stack(sent_level_encoder, word_level_encoder,masking=True)
                elif config.use_sent_level and not config.use_word_level: # SL
                    dec_outputs = sent_level_encoder
                elif config.use_word_level and not config.use_sent_level: #WL
                    dec_outputs = encoder

        with tf.variable_scope("network_scores"): #Network score
            net_scores = tf.contrib.layers.fully_connected(
            dec_outputs,
            config.ntags,
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=config.l1_regul, scale_l2=config.l2_regul),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="network_scores_layer")

        with tf.variable_scope("sentence_score"):
            sentence_score = net_scores

        with tf.variable_scope("prediction"):
            self.logits = tf.reshape(sentence_score, (-1, tf.shape(self.word_ids)[1], config.ntags))

        #loss layer
        with tf.variable_scope("loss"):
            """Defines the loss"""
            if self.config.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.seqlens)
                self.trans_params = trans_params # need to evaluate it for decoding
                self.loss = tf.reduce_mean(-log_likelihood)
            else :
                # model predictions
                self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
                mask =  tf.sequence_mask(self.seqlens,maxlen=self._maxlen)
                mask = tf.cast(mask,dtype= tf.float32)
                losses = tf.math.multiply(mask,losses)
                self.loss = tf.reduce_mean(losses)

            # for tensorboard
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('accuracy'):
            # evaluate model
            correct_pred = tf.equal(tf.cast(tf.argmax(self.logits,-1),tf.int32),self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='num_correct')

        # provide accuracy information
        tf.summary.scalar('accuracy', self.accuracy)







