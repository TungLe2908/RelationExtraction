from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input, Embedding, Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import TimeDistributed, Flatten, Permute,Activation, RepeatVector, Lambda, Reshape
from keras.layers import Add, Multiply, Subtract, Dot, Concatenate, merge, Average
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import regularizers
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention, MultiHead
import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cPickle
import warnings
warnings.filterwarnings('ignore') 

def evaluate_score(y_true, y_pred):
	y_true_label = (np.argmax(y_true, axis = 1)).flatten()
	y_pre_label = (np.argmax(y_pred, axis = 1)).flatten()
	#print(y_true_label.shape)
	#print(y_pre_label[:100])
	pre = precision_score(y_true_label, y_pre_label, average = 'macro')
	rec = recall_score(y_true_label, y_pre_label, average = 'macro')
	f1 = f1_score(y_true_label, y_pre_label, average = 'macro')
	return pre, rec, f1

	
x1 = cPickle.load(open("all_glove_dic.p", "rb"))
embedding_matrix_glove = x1[0]

x2 = cPickle.load(open("all_fasttext_dic.p", "rb"))
embedding_matrix_fasttext = x2[0]

x3 = cPickle.load(open("all_wordnet_dic.p", "rb"))
embedding_matrix_wordnet = x3[0]

shape_embed = embedding_matrix_glove.shape
EMBEDDING_DIM = 300
VOCAB_ENCODER = shape_embed[0]

# model define
model = Sequential()
	# encoder LSTM
input_timestep = 19
encoder_dim = 50
position_dim = 50
num_classes = 10

#input

words1_inputs = Input(shape=(input_timestep,))
words2_inputs = Input(shape=(input_timestep,))

relation_inputs = Input(shape=(input_timestep,))

dis1_w1_in = Input(shape=(input_timestep,))
dis2_w1_in = Input(shape=(input_timestep,))
dis1_w2_in = Input(shape=(input_timestep,))
dis2_w2_in = Input(shape=(input_timestep,))

#embedding

embed_layer_word_glove = Embedding(VOCAB_ENCODER, EMBEDDING_DIM, 
										weights=[embedding_matrix_glove],trainable=False)

words1_embedding_glove = embed_layer_word_glove(words1_inputs)
words2_embedding_glove = embed_layer_word_glove(words2_inputs)

embed_layer_word_fasttext = Embedding(
							VOCAB_ENCODER, EMBEDDING_DIM, 
							weights=[embedding_matrix_fasttext],trainable=False)

words1_embedding_fasttext = embed_layer_word_fasttext(words1_inputs)
words2_embedding_fasttext = embed_layer_word_fasttext(words2_inputs)


embed_layer_word_wordnet = Embedding(VOCAB_ENCODER, EMBEDDING_DIM, 
										weights=[embedding_matrix_wordnet],trainable=False)

words1_embedding_wordnet = embed_layer_word_wordnet(words1_inputs)
words2_embedding_wordnet = embed_layer_word_wordnet(words2_inputs)

no_relation_max = 41
output_rel_dim = 900
embed_layer_relation = Embedding(no_relation_max, output_rel_dim)

relation_embedding = embed_layer_relation(relation_inputs)

no_pos_max = 175
output_pos_dim = 50
#embed_position_layer =  Embedding(no_pos_max, output_pos_dim)
pos1_w1 = Reshape((input_timestep,1))(dis1_w1_in)#embed_position_layer(dis1_w1_in)
pos2_w1 = Reshape((input_timestep,1))(dis2_w1_in)#embed_position_layer(dis2_w1_in)
pos1_w2 = Reshape((input_timestep,1))(dis1_w2_in)#embed_position_layer(dis1_w2_in)
pos2_w2 = Reshape((input_timestep,1))(dis2_w2_in)#embed_position_layer(dis2_w2_in)


#model

self_w11_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),
								attention_regularizer_weight=1e-4, name='Attention11')

self_w12_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention12')

self_w21_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention21')


self_w22_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention22')
								
self_w31_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention31')


self_w32_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention32')								

self_rel_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention3')
								
word1_glove = self_w11_att_layer(words1_embedding_glove)
word2_glove = self_w12_att_layer(words2_embedding_glove)

word1_fasttext = self_w21_att_layer(words1_embedding_fasttext)
word2_fasttext = self_w22_att_layer(words2_embedding_fasttext)

word1_wordnet = self_w31_att_layer(words1_embedding_wordnet)
word2_wordnet = self_w32_att_layer(words2_embedding_wordnet)

words1 = Concatenate()([word1_glove, word1_fasttext, word1_wordnet])
words2 = Concatenate()([word2_glove, word2_fasttext, word2_wordnet])
relation = self_rel_att_layer(relation_embedding)
		
				
info = Average()([words1,relation, words2])

info = MultiHeadAttention(head_num=3,name='Multi-Head')([info, info,info])

info = LSTM(units=900)(info)


dense_layer = Dense(300, activation = 'sigmoid')
w = dense_layer(info)

prob_layer = Dense(num_classes, activation = 'softmax')
prob = prob_layer(w)

model = Model(
		inputs = [words1_inputs, words2_inputs, relation_inputs, 
					dis1_w1_in, dis2_w1_in, dis1_w2_in, dis2_w2_in], 

		outputs = prob
		
		)

model.summary()

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['acc'])

filepath="weights_lstm_3wordembedd_multihead.hdf5"
monitor_point = 'val_acc'
checkpointer = ModelCheckpoint(filepath, monitor=monitor_point, verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor=monitor_point, patience=100)
#### data 

train = cPickle.load(open("train_dep.p", "rb"))
train_y = train[0]
train_x1 = train[1] #w1
train_x2 = train[2] #w2
train_x3 = train[3] #rel
train_x4 = train[4] #d1w1
train_x5 = train[5] #d2w1
train_x6 = train[6] #d1w2
train_x7 = train[7] #d2w2

test = cPickle.load(open("test_dep.p", "rb"))
test_y = test[0]
test_x1 = test[1]
test_x2 = test[2]
test_x3 = test[3]
test_x4 = test[4]
test_x5 = test[5]
test_x6 = test[6]
test_x7 = test[7]


model.load_weights(filepath)
history = model.fit([train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7], 
						batch_size=50, 
						y=to_categorical(train_y, num_classes), 
						validation_data=([test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7],
						to_categorical(test_y, num_classes)), 
						shuffle=True, epochs=500, callbacks=[checkpointer, early_stop])

### test
'''
for k in range(5):
	for i in range(1,8,1):
		print(train[i][k])
'''


model.load_weights(filepath)
loss, acc = model.evaluate(x = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7], 
							y = to_categorical(test_y, num_classes))
print acc

y_true = to_categorical(test_y, num_classes)

y_pred = model.predict(x = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7])



print y_pred[:5]
print test_y[:5]



pre, rec, f1 = evaluate_score(y_true, y_pred)

print pre
print rec
print f1

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

loss = cross_entropy(y_pred[:5], y_true[:5])
print(loss)




