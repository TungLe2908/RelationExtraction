import numpy as np
import csv
import nltk
from gensim.models.keyedvectors import KeyedVectors
import cPickle
from keras.preprocessing.sequence import pad_sequences
import BeautifulSoup

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def convert2id(word, word2id, id2word, index):
	word = word.lower()
	if not(word in word2id):
		word2id[word] = index
		id2word[index] = word
		index += 1
		
	return word2id[word], word2id, id2word, index

	
def get_label(label, list_label):
	label = label.split()[0]
	if (label == 'Other'):
			label_index = 0
	else: 
		pos_label = label.index('(')
		label_index = list_label.index(label[0:pos_label])
	return label_index

def get_tag_content(content, tag):
	#tag = e1 or e2
	n = len(content)
	list_content = []
	i = 0
	list_content_idx = 0
	o_tag = '<'+ tag + '>'
	c_tag = '</' + tag + '>'
	e = ''
	while (i < n):
		if (o_tag in content[i]):
			index_e = list_content_idx
			content[i] = content[i].replace(o_tag,'')
			if (c_tag in content[i]):
				#1 word
				e = content[i].replace(c_tag,'')
			else:
				e = content[i]
				i = i + 1
				while ((c_tag in content[i]) == False):
					e = e + ' ' + content[i]
					i = i + 1
				e = e + ' ' + content[i].replace(c_tag,'')
			list_content.append(e)
			list_content_idx += 1
		else:
			list_content.append(content[i])
			list_content_idx += 1
		i = i + 1
	return list_content, e
	
def get_content(content):
	content = content.split('\t')[1]
	content = content.replace('"', '')
	content = content.replace('.', ' .')
	
	nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')
	
	content = nlp.word_tokenize(content)
	
	nlp.close()
	
	print('finish tokenize')
	
	list_content, e1 = get_tag_content(content, 'e1')
	list_content, e2 = get_tag_content(list_content, 'e2')
	
	index_e1 = list_content.index(e1)
	index_e2 = list_content.index(e2)
	
	tmp_e1_idx = -index_e1
	tmp_e2_idx = -index_e2
	pad_e1 = []
	pad_e2 = []
	
	for i in range(len(list_content)):	
		pad_e1.append(tmp_e1_idx)
		pad_e2.append(tmp_e2_idx)
		tmp_e1_idx += 1
		tmp_e2_idx += 1
	
	return [list_content, pad_e1, pad_e2, index_e1, index_e2] 


def check_dep(bound, dep_tree):
	for ele in dep_tree:
		if (ele[1] > bound or ele[2] > bound): 
			print ele[1], ele[2]
			return False
	return True
	


from stanfordcorenlp import StanfordCoreNLP	
def read_semeval(path_in, word2id, id2word, index):
	file = open(path_in,'r')
	content = file.readlines()
	
	list_label = ['Other', 'Component-Whole', 'Cause-Effect', 
						'Entity-Destination', 'Member-Collection', 
						'Message-Topic', 'Entity-Origin', 'Product-Producer', 
						'Content-Container', 'Instrument-Agency']
	n = len(content)
	sent_id_list = []
	pad_e1_list = []
	pad_e2_list = []
	label_list = []
	dep_tree_list = []
	length = 0
	no_sent = 0
	
	nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')
	
	for i in range(0,n,4):
		sent = content[i]
		label = content[i+1]
		com = content[i+2] 
		label_index = get_label(label, list_label)
		
		sent_cont, pad_e1, pad_e2, idx_e1, idx_e2 = get_content(sent)
		
		
		# Get Dependency Tree
		sent_full = " ".join(sent_cont)
		sent_dep_tree = nlp.dependency_parse(sent_full)
		dep_tree_list.append(sent_dep_tree)
		
		
		sent_id = []
		for ele in sent_cont:
			ele_idx, word2id, id2word, index = convert2id(ele, word2id, id2word, index)
			sent_id.append(ele_idx)
		
		'''
		if (check_dep(len(sent_id) +1, sent_dep_tree) == False ): 
			print 'Arlet'
			print len(sent_cont)
		'''

		label_list.append(label_index)
		pad_e1_list.append(pad_e1)
		pad_e2_list.append(pad_e2)
		sent_id_list.append(sent_id)
		
		length += len(sent_id)
		no_sent +=1
		
	nlp.close()	
	print no_sent, length, length/no_sent
	return [label_list, pad_e1_list, pad_e2_list, sent_id_list, dep_tree_list], word2id, id2word, index

def readBaroni(path):
	model_org = {}
	with open(path) as f:
		for line in f:
			row = line.split(' ')
			model_org[row[0]] = np.array(row[1:])
	print len(model_org["you"])
	return model_org

def readBaroniO(path):
	model_org = {}
	with open(path) as f:
		for line in f:
			row = line.split('\t')
			model_org[row[0]] = np.array(row[1:])
	print len(model_org["you"])
	return model_org

def saveW(word2vec_path, bin, dic_name, word2id, id2word):
	#global word2id
	#global id2word
	dimension = 300
	if word2vec_path == "../WordEmbedding/EN-wform.w.5.cbow.neg10.400.subsmpl.txt":
		model_org = readBaroniO(word2vec_path)
		dimension = 400
	elif word2vec_path == 'glove.840B.300d.txt':
		model_org = readBaroni(word2vec_path)
	elif word2vec_path == "../WordEmbedding/paragram_300_sl999/paragram_300_sl999.txt":
		model_org = readBaroni(word2vec_path)
	else:
		model_org = KeyedVectors.load_word2vec_format(word2vec_path, binary=bin)
			
	#get W weight for embedding layer

	W = np.zeros(shape=(len(word2id)+1+2, dimension), dtype='float32')
	W[0] = np.zeros(dimension, dtype='float32')

	count_in = 0
	count_out = 0
	flag = True
	for word in word2id:
		i = word2id[word]
		if (len(word) > 1):
			temp_W = np.zeros(dimension, dtype='float32')
			words = word.split()
			for ele in words:
				if ele in model_org:
					word_vec = model_org[ele]
					temp_W1 = np.add(temp_W,word_vec.astype('float32'))
					temp_W = temp_W1
				else:
					flag = False
					temp_W1 = np.add(temp_W,np.random.uniform(-0.25,0.25,dimension))
					temp_W = temp_W1
			W[i] = temp_W
			if (flag == False):
				count_out += 1
				
			else:
				count_in += 1
			
			flag = True
		else:
			if word in model_org:
				W[i] = model_org[word]
				count_in += 1
			else:
				W[i] = np.random.uniform(-0.25,0.25,dimension)
				count_out += 1

	print (dic_name,count_in, count_out)
	print W.shape
	cPickle.dump([W,word2id,id2word], open(dic_name, "wb"))
	
	
def main_process(word2id, id2word, index, path_in, path_out, pad_mode = True):
	
	print('Start ' + path_out)
	
	package, word2id, id2word, index = read_semeval(path_in, word2id, id2word, index)
	labels, pad_e1s, pad_e2s, sentids, dep_trees = package[0], package[1], package[2], package[3], package[4]
	if (pad_mode == True):
		pad_e1s = pad_sequences(pad_e1s, 25 , value = 25 , padding = 'post',truncating = 'post')
		pad_e2s = pad_sequences(pad_e2s, 25 , value = 25 , padding = 'post',truncating = 'post')
		sentids = pad_sequences(sentids, 25 , value = 0 , padding = 'post',truncating = 'post')
	
	cPickle.dump([labels, sentids, dep_trees, pad_e1s, pad_e2s], open(path_out, "wb"))
	
	print('Finish ' + path_out)
	
	

def get_ele_pos(arr, item):
	for i in range(len(arr)):
		if (arr[i] == item): return i
	return -1

def save_Dep_rel(rel2index, index, path_in):
	
	x = cPickle.load(open(path_in,'r'))
	labels,sentids,  dep_trees, pad_e1s, pad_e2s = x[0], x[1], x[2], x[3], x[4], 
	for ele in dep_trees:
		for triple in ele:
			relation = triple[0]
			
			if not(relation in rel2index):
				rel2index[relation] = index
				index += 1
	name = 'rel2index.p'
	cPickle.dump([rel2index], open(name, "wb"))

def save_new_format(path_in, path_out):
	
	x = cPickle.load(open(path_in,'r'))
	labels,sentids, dep_trees, pad_e1s, pad_e2s = x[0], x[1], x[2], x[3], x[4]
	
	rel2index = (cPickle.load(open('rel2index.p','r')))[0]
	
	max_length = 96
	count = 0
	n_sample = len(labels)
	w1s = []
	w2s = []
	rels = []
	dis1_w1s = []
	dis2_w1s = []
	dis1_w2s = []
	dis2_w2s = []
	min_dis1 = 10
	min_dis2 = 10
	max_dis1 = 0
	max_dis2 = 0
	for i in range(n_sample):
		flag = False
		sentid = sentids[i]
		dep_tree = dep_trees[i]
		
		pad_e1 = [x+75 for x in pad_e1s[i]] 
		pad_e2 = [x+75 for x in pad_e2s[i]] 
		
		if (min_dis1 > min(pad_e1)): min_dis1 = min(pad_e1)
		if (min_dis2 > min(pad_e2)): min_dis2 = min(pad_e2)
		if (max_dis1 < max(pad_e1)): max_dis1 = max(pad_e1)
		if (max_dis2 < max(pad_e2)): max_dis2 = max(pad_e2)
		w1 = []
		w2 = []
		rel = []
		dis1_w1 = []
		dis2_w1 = []
		dis1_w2 = []
		dis2_w2 = []
		#for each tuple
		for j in range(1, len(dep_tree)):
			triple = dep_tree[j]
			try: 
				rel.append(rel2index[triple[0]])
				w1.append(sentid[triple[1] - 1])
				w2.append(sentid[triple[2] - 1])
				dis1_w1.append(pad_e1[triple[1] - 1])
				dis2_w1.append(pad_e2[triple[1] - 1])
				dis1_w2.append(pad_e1[triple[2] - 1])
				dis2_w2.append(pad_e2[triple[2] - 1])
			except: 
				flag = True
				
		w1s.append(w1)
		w2s.append(w2)
		rels.append(rel)
		dis1_w1s.append(dis1_w1)
		dis2_w1s.append(dis2_w1)
		dis1_w2s.append(dis1_w2)
		dis2_w2s.append(dis2_w2)
		if (flag == True):
			count += 1
	#padding
	pad_w1s = pad_sequences(w1s, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_w2s = pad_sequences(w2s, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_rels = pad_sequences(rels, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_dis1_w1s = pad_sequences(dis1_w1s, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_dis2_w1s = pad_sequences(dis2_w1s, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_dis1_w2s = pad_sequences(dis1_w2s, 19 , value = 0 , padding = 'post',truncating = 'post')
	pad_dis2_w2s = pad_sequences(dis2_w2s, 19 , value = 0 , padding = 'post',truncating = 'post')
	
	cPickle.dump([labels,pad_w1s, pad_w2s, pad_rels, pad_dis1_w1s, pad_dis2_w1s, pad_dis1_w2s, pad_dis2_w2s], open(path_out, "wb"))
	
	print('min_d1: '+ str(min_dis1) + ' min_d2:' + str(min_dis2) )
	print('max_d1: '+ str(max_dis1) + ' max_d2:' + str(max_dis2) )
	print('Finish ' + path_out )
	print('Errors:' + str(count) + ' / ' + str(len(labels)))
	
		
#f = open("demo.txt","r")
#contents = f.readlines()

word2id = {}
id2word = {}
entityword = {}
index = 1

path_train = path_train = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt'
path_test = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt'

#main_process(word2id, id2word, index, path_train, 'train_no_pad.p', False)
#main_process(word2id, id2word, index, path_test, 'test_no_pad.p', False)

#path_word2vec = 'wiki-news-300d-1M.vec'
#saveW(path_word2vec, False,'all_fasttext_dic.p', word2id, id2word)

save_new_format('train_no_pad.p','train_dep.p')
save_new_format('test_no_pad.p','test_dep.p')








