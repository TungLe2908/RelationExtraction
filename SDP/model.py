from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input, Embedding, Conv1D, MaxPooling1D, merge
from keras.layers import TimeDistributed, Flatten, Permute,Activation, RepeatVector, Lambda, Reshape
from keras.layers import Add, Multiply, Subtract, Dot, Concatenate
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Layer, InputSpec
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from keras_self_attention import SeqSelfAttention
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

def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed
	

def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR = False ):
    # inputs.shape = (batch_size, time_steps, input_dim)
	input_dim = int(inputs.shape[2])
	TIME_STEPS = int(inputs.shape[1])
	a = Permute((2, 1))(inputs)
	print TIME_STEPS
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
	a = Dense(TIME_STEPS, activation='softmax', name = 'attention_dense_3d')(a)
	if SINGLE_ATTENTION_VECTOR:
		a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
		a = RepeatVector(input_dim)(a)
	a_probs = Permute((2, 1), name='attention_vec_3d')(a)
	output_attention_mul = merge([inputs, a_probs], mode='mul')
	return output_attention_mul

def attention_2d_block(inputs,input_dim):
	attention_probs = Dense(input_dim, activation='softmax')(inputs)
	attention_mul = merge([inputs, attention_probs], output_shape=32, mode='mul')
	return attention_mul
	
x = cPickle.load(open("all_fasttext_dic.p", "rb"))
embedding_matrix = x[0]

shape_embed = embedding_matrix.shape
	
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

embed_layer_word = Embedding(VOCAB_ENCODER, EMBEDDING_DIM, weights=[embedding_matrix],trainable=False)
words1_embedding = embed_layer_word(words1_inputs)
words2_embedding = embed_layer_word(words2_inputs)

no_relation_max = 41
output_rel_dim = 100
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
'''
self_w1_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),
								attention_regularizer_weight=1e-4, name='Attention1')

self_w2_att_layer = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
								kernel_regularizer=keras.regularizers.l2(1e-4), 
								bias_regularizer=keras.regularizers.l1(1e-4),
								attention_regularizer_weight=1e-4, name='Attention2')


words1 = self_w1_att_layer(words1_embedding)
words2 = self_w2_att_layer(words2_embedding)
'''

words1 = words1_embedding
words2 = words2_embedding

rd_dim = 500
rd_embed = Concatenate()([pos1_w1,pos1_w2,relation_embedding, pos2_w1,pos2_w2])
rd_layer = Dense(rd_dim, activation = 'sigmoid')
rd_embed = rd_layer(rd_embed)

	#combine Rd and word1, word2

W1rd = Dense(rd_dim)

W2rd = Dense(rd_dim)

w1 = Multiply()([W1rd(words1), rd_embed])
w2 = Multiply()([W2rd(words1), rd_embed])

w1 = Activation('sigmoid')(w1)
w2 = Activation('sigmoid')(w2)

w = Concatenate()([w1,w2])


#CNN classification
cnn_layer1 = Conv1D(filters=500, kernel_size=3, activation='relu')
cnn_layer2 = Conv1D(filters=250, kernel_size=4, activation='relu')
cnn_layer3 = Conv1D(filters=100, kernel_size=5, activation='relu')
maxpooling_layer1 = MaxPooling1D(pool_size=17)
maxpooling_layer2 = MaxPooling1D(pool_size=16)
maxpooling_layer3 = MaxPooling1D(pool_size=15)

w1 = cnn_layer1(w)
w1 = maxpooling_layer1(w1)

w2 = cnn_layer2(w)
w2 = maxpooling_layer2(w2)

w3 = cnn_layer3(w)
w3 = maxpooling_layer3(w3)

w1 = Flatten()(w1)
w2 = Flatten()(w2)
w3 = Flatten()(w3)

w = Concatenate()([w1,w2,w3])

dense_layer = Dense(400, activation = 'sigmoid')
w = dense_layer(w)

prob_layer = Dense(num_classes, activation = 'softmax')
prob = prob_layer(w)

model = Model(
		inputs = [words1_inputs, words2_inputs, relation_inputs, 
					dis1_w1_in, dis2_w1_in, dis1_w2_in, dis2_w2_in], 

		outputs = prob
		
		)

model.summary()

opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['acc'])

filepath="weights_dt_attention.hdf5"
monitor_point = 'val_acc'
checkpointer = ModelCheckpoint(filepath, monitor=monitor_point, verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor=monitor_point, patience=25)
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


history = model.fit([train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7], 
						batch_size=50, 
						y=to_categorical(train_y, num_classes), 
						validation_data=([test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7],
						to_categorical(test_y, num_classes)), 
						shuffle=True, epochs=100, callbacks=[checkpointer, early_stop])

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



'''

## Evaluate
from sklearn.metrics import precision_recall_curve

def getdata(a, pos):
	result = []
	for ele in pos:
		result.append(a[ele])
	return np.array(result)
	
def score_atN_average(model, X, Y, n):
	x1 = X[0]
	x2 = X[1]
	x3 = X[2]
	pos = np.arange(len(Y))
	np.random.shuffle(pos)
	i = 0
	ave_pre = 0
	ave_rec = 0
	ave_f1 = 0
	time = 0
	while( i < len(Y)):
		a = pos[i: i + n]
		test_x1 = getdata(x1,a)
		test_x2 = getdata(x2,a)
		test_x3 = getdata(x3,a)
		test_y = getdata(Y,a)
		y_true = to_categorical(test_y, num_classes)
		y_pred = model.predict(x = [test_x1,test_x2,test_x3])
		pre, rec, f1 = evaluate_score(y_true, y_pred)
		i = i + n
		ave_pre += pre
		ave_rec += rec
		ave_f1 += f1
		time += 1
	return ave_pre/time, ave_rec/time, ave_f1/time	

def score_atN_max(model, X, Y, n):
	x1 = X[0]
	x2 = X[1]
	x3 = X[2]
	pos = np.arange(len(Y))
	np.random.shuffle(pos)
	i = 0
	pre_max = 0
	rec_max = 0
	f1_max = 0
	time = 0
	while( i < len(Y)):
		a = pos[i: i + n]
		test_x1 = getdata(x1,a)
		test_x2 = getdata(x2,a)
		test_x3 = getdata(x3,a)
		test_y = getdata(Y,a)
		y_true = to_categorical(test_y, num_classes)
		y_pred = model.predict(x = [test_x1,test_x2,test_x3])
		pre, rec, f1 = evaluate_score(y_true, y_pred)
		i = i + n
		if (f1 > f1_max): 
			pre_max = pre
			rec_max = rec
			f1_max = f1
		
	return pre_max, rec_max, f1_max	
	
	
def score_atN_random(model, X, Y, n):
	x1 = X[0]
	x2 = X[1]
	x3 = X[2]
	pos = np.arange(len(Y))
	np.random.shuffle(pos)
	a = pos[: n]
	test_x1 = getdata(x1,a)
	test_x2 = getdata(x2,a)
	test_x3 = getdata(x3,a)
	test_y = getdata(Y,a)
	y_true = to_categorical(test_y, num_classes)
	y_pred = model.predict(x = [test_x1,test_x2,test_x3])
	pre, rec, f1 = evaluate_score(y_true, y_pred)
		
	return pre, rec, f1		

N = [1,5,10,20,50, 100, 200, 300]
print('For MAX N')

for ele in N:
	print ('With ' + str(ele))	
	pre,rec,f1 = score_atN_max(model, [test_x1, test_x2, test_x3], test_y, ele)	
	print pre
	print rec
	print f1 
	
print('For RANDOM N')	
for ele in N:
	print ('With ' + str(ele))	
	pre,rec,f1 = score_atN_random(model, [test_x1, test_x2, test_x3], test_y, ele)	
	print pre
	print rec
	print f1 
	
print('For AVERAGE N')
for ele in N:
	print ('With ' + str(ele))	
	pre,rec,f1 = score_atN_average(model, [test_x1, test_x2, test_x3], test_y, ele)	
	print pre
	print rec
	print f1 
'''

