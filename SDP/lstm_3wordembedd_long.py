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
'''	
x2 = cPickle.load(open("all_fasttext_dic.p", "rb"))
embedding_matrix_fasttext = x2[0]

x3 = cPickle.load(open("all_wordnet_dic.p", "rb"))
embedding_matrix_wordnet = x3[0]
'''	


shape_embed = embedding_matrix_glove.shape
EMBEDDING_DIM = 300
VOCAB_ENCODER = shape_embed[0]

# model define
model = Sequential()
	# encoder LSTM
input_timestep = 70
encoder_dim = 50
position_dim = 50
num_classes = 10

#input

sent_inputs = Input(shape=(input_timestep,))
words1_inputs = Input(shape=(input_timestep,))
words2_inputs = Input(shape=(input_timestep,))

relation_inputs = Input(shape=(input_timestep,))
ner_w1_inputs = Input(shape=(input_timestep,))
ner_w2_inputs = Input(shape=(input_timestep,))
pos_w1_inputs = Input(shape=(input_timestep,))
pos_w2_inputs = Input(shape=(input_timestep,))

#embedding

embed_layer_word_glove = Embedding(VOCAB_ENCODER, EMBEDDING_DIM, 
										weights=[embedding_matrix_glove],trainable=True)


sent_embed = embed_layer_word_glove(sent_inputs) 										
words1_embedding = embed_layer_word_glove(words1_inputs)
words2_embedding = embed_layer_word_glove(words2_inputs)

no_ner_max = 100
output_ner_dim = 50
embed_layer_ner = Embedding(no_ner_max, output_ner_dim, embeddings_initializer='uniform')
ner_w1_embed = embed_layer_ner(ner_w1_inputs)
ner_w2_embed = embed_layer_ner(ner_w2_inputs)


no_pos_max = 100
output_pos_dim = 50
embed_layer_pos = Embedding(no_pos_max, output_pos_dim, embeddings_initializer='uniform')
pos_w1_embed = embed_layer_pos(pos_w1_inputs)
pos_w2_embed = embed_layer_pos(pos_w2_inputs)

no_relation_max = 41
output_rel_dim = 400
embed_layer_relation = Embedding(no_relation_max, output_rel_dim, embeddings_initializer='uniform')

relation_embedding = embed_layer_relation(relation_inputs)


#model

self_w1_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),
								attention_regularizer_weight=1e-4)

self_w2_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)

self_pos1_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)

self_pos2_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)


self_ner1_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)

self_ner2_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)
							
self_rel_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)

self_sent_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4)


ner_w1 = self_ner1_att_layer(ner_w1_embed)
ner_w2 = self_ner2_att_layer(ner_w2_embed)
pos_w1 = self_pos1_att_layer(pos_w1_embed)
pos_w2 = self_pos2_att_layer(pos_w2_embed)

								
word1 = self_w1_att_layer(words1_embedding)
word2 = self_w2_att_layer(words2_embedding)

words1 = Concatenate()([word1, ner_w1, pos_w1]) #400
words2 = Concatenate()([word2, ner_w2, pos_w2]) #400

relation = self_rel_att_layer(relation_embedding)
					
tuple = Average()([words1,relation, words2])

#info = MultiHeadAttention(head_num=2,name='Multi-Head')([words1, relation,words2])

tuple = LSTM(units=500)(tuple)

sent = self_sent_att_layer(sent_embed)
sent = LSTM(units = 500)(sent)

info = Average()([sent, tuple])

dense_layer = Dense(300, activation = 'sigmoid')
prob_layer = Dense(num_classes, activation = 'softmax')
w1 = dense_layer(info)
prob = prob_layer(w1)

dense_layer2 = Dense(300, activation = 'sigmoid')
prob_layer2 = Dense(num_classes, activation = 'softmax')
w2 = dense_layer2(sent)
prob2 = prob_layer2(w2)


dense_layer3 = Dense(300, activation = 'sigmoid')
prob_layer3 = Dense(num_classes, activation = 'softmax')
w3 = dense_layer3(tuple)
prob3 = prob_layer3(w3)


prob = Average()([prob, prob2, prob3])

model = Model(
		inputs = [sent_inputs, words1_inputs, words2_inputs, relation_inputs, 
					ner_w1_inputs, ner_w2_inputs, pos_w1_inputs, pos_w2_inputs], 

		outputs = prob
		
		)

model.summary()

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['acc'])

filepath="weights_lstmlong.hdf5"
monitor_point = 'val_acc'
checkpointer = ModelCheckpoint(filepath, monitor=monitor_point, verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor=monitor_point, patience=100)
#### data 

train = cPickle.load(open("train_fullinfo_dep.p", "rb"))
train2 = cPickle.load(open("train_fulletf.p", "rb"))
sent_train = train2[1]
train_y = train[0]
train_x1 = train[1] #w1
train_x2 = train[2] #w2
train_x3 = train[3] #rel
train_x4 = train[4] #d1w1
train_x5 = train[5] #d2w1
train_x6 = train[6] #d1w2
train_x7 = train[7] #d2w2
train_x8 = train[8] #nerw1
train_x9 = train[9] #nerw2
train_x10 = train[10] #posw1
train_x11 = train[11] #posw2

test = cPickle.load(open("test_fullinfo_dep.p", "rb"))
test2 = cPickle.load(open("test_fulletf.p", "rb"))
sent_test = test2[1]
test_y = test[0]
test_x1 = test[1]
test_x2 = test[2]
test_x3 = test[3]
test_x4 = test[4]
test_x5 = test[5]
test_x6 = test[6]
test_x7 = test[7]
test_x8 = test[8]
test_x9 = test[9]
test_x10 = test[10]
test_x11 = test[11]


model.load_weights(filepath)
history = model.fit([sent_train, train_x1, train_x2, train_x3, train_x8, train_x9, train_x10, train_x11 ], 
						batch_size=50, 
						y=to_categorical(train_y, num_classes), 
						validation_data=([sent_test, test_x1, test_x2, test_x3, test_x8, test_x9, test_x10, test_x11],
						to_categorical(test_y, num_classes)), 
						shuffle=True, epochs=50, callbacks=[checkpointer, early_stop])

### test
'''
for k in range(5):
	for i in range(1,8,1):
		print(train[i][k])
'''


model.load_weights(filepath)
loss, acc = model.evaluate(x = [sent_test, test_x1, test_x2, test_x3, test_x8, test_x9, test_x10, test_x11], 
							y = to_categorical(test_y, num_classes))
print acc

y_true = to_categorical(test_y, num_classes)

y_pred = model.predict(x = [sent_test, test_x1, test_x2, test_x3, test_x8, test_x9, test_x10, test_x11])



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




