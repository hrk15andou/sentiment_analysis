# Information Gain Module
# Update : 2017/01/01(Sun)

# ---- Import modules -------------------------------------- #

import numpy as np

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

