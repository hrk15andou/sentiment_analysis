# Support Vector Sentiment Analysis
# Information Gain (IG) Model
# Update : 2016/12/08(Tue)

# ---- Import modules -------------------------------------- #

import os
import csv
import glob
import numpy as np
import numpy
import MeCab
from gensim import corpora, models, similarities, matutils
from sklearn import svm, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


# ---- Read imdb ------------------------------------------- #

def load(num):
	flag = [] # label
	doc = []  # documents
	os.chdir("./pos") # change directory
	n=0
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			flag.append(1)
			doc.append(f.readline().strip('\n'))
		n=n+1
		if n >= 10000:
			break
	n=0
	os.chdir("..")
	os.chdir("./neg")
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			flag.append(0)
			doc.append(f.readline().rstrip('\n'))
		n=n+1
		if n >= 10000:
			break

	metag = MeCab.Tagger("-Owakati")

	jpn=[]
	for sent in doc:
		jpn.append(metag.parse(sent).rstrip('\n'))

	stoplist = set('！ 。 、 ？ （ ）「 」 ( ) ! ? , .'.split())
	texts = [[word for word in t.split() if word not in stoplist]for t in jpn]

	dic = corpora.Dictionary(texts)
	#dic.filter_extremes(no_below=20, no_above=0.3)

	vector = []

	tmp = [dic.doc2bow(d) for d in texts]

	tfidf_model = models.TfidfModel(tmp)
	tfidf_corpus = tfidf_model[tmp]

	lsi_model = models.LsiModel(tfidf_corpus, id2word=dic, num_topics=num)
	lsi_corpus = lsi_model[tfidf_corpus]

	for txt in lsi_corpus :
		vector.append(list(matutils.corpus2dense([txt], num_terms=num).T[0]))

	vec_t, vec_s, flg_t, flg_s = train_test_split(vector,flag, test_size=0.4)

	return(vec_t,flg_t,vec_s,flg_s)

# ---- Calculate Information Gain -------------------------- #

def e(x, y):
    return - (1.0 * x / (x + y)) * np.log2(1.0 * x / (x + y)) - (1.0 * y / (x + y)) * np.log2(1.0 * y / (x + y))

def ig(word, data, category):
	total = len(data)
	tp = np.sum([word in data[i] for i in range(0, len(data)) if category[i] == 1])
	fp = np.sum([word in data[i] for i in range(0, len(data)) if category[i] == 0])
	pos = np.sum(category)
	neg = len(data) - pos
	fn = pos - tp
	tn = neg - fp
	return e(pos, neg) - (1.0 * (tp + fp) / total * e(tp, fp) + (1.0 - 1.0 * (tp + fp) / total) * e(fn, tn))



# ---- Learning by SVM  ------------------------------------ #

def mlearn(docs,flags,tdocs,tflgs,num):

	'''
	print("--- Testing Data ---------------")
	print("Training Data : ",end='')
	print(len(flags))
	print("Test Data     : ",end='')
	print(len(tflgs))
	print("--------------------------------")
	'''

	parameters = [{'kernel':['rbf'], 'gamma':[1e-3, 1e-4],
			'C': [1, 10, 100, 1000]},
			{'kernel':['linear'], 'C':[1,10,100,1000]}]

	svc = grid_search.GridSearchCV(svm.SVC(), parameters, cv=5, n_jobs = -1)

	svc.fit(docs,flags)
	y_true, y_pred = tflgs, svc.predict(tdocs)
	print(classification_report(y_true, y_pred))

	data = []
	res = classification_report(y_true, y_pred).split(' ')
	for i in res:
		if i != '':
			data.append(i.rstrip().encode('utf-8'))

	writers = []
	writers.append(num)
	writers.append(float(data[5]))
	writers.append(float(data[6]))
	writers.append(float(data[10]))
	writers.append(float(data[11]))
	writers.append(float(data[17]))

	return writers


# ---- Main Function  -------------------------------------- #

def main():	
	i = 50
	p = []
	tag = []
	for i in range(50,150,50):
		os.chdir("/Users/lab/Desktop/svm/jpn")
		print(i)
		docs,flags,tdocs,tflgs = load(i)

		p.append(mlearn(docs,flags,tdocs,tflgs,i))

	print(p)

	os.chdir("/Users/lab/Desktop/svm/jpn")

	with open('data.csv', 'w') as f:
		print('open')
		writecsv = csv.writer(f)
		writecsv.writerow(['num','Pre(neg)','Rec(neg)','Pre(pos)','Rec(pos)','Acc'])
		for q in p:
			writecsv.writerow(q)

	print('finish')

if __name__ == '__main__':
	main()
