# Sentiment Analyzer Mainframe
# Update : 2017/01/01(Sun)

# ---- Import modules -------------------------------------- #

import os
import csv

# ---- Main Function  -------------------------------------- #

def main():
	p = []
	tag = []

	tfidf_corpus,dic,flag=load()

	for i in range(50,150,50):
		os.chdir("/Users/lab/Desktop/svm/jpn")
		print(i)
		docs,flags,tdocs,tflgs = lsi(tfidf_corpus,dic,flag,i)

		mlearn(docs,flags,tdocs,tflgs,i)

	print('finish')

if __name__ == '__main__':
	main()
