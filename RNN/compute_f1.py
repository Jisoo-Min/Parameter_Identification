
def compute_score():
	filepath = './elman-forward/current.test.txt'
	from collections import Counter
	from sklearn.metrics import classification_report

	cnt = 0 
	equal = 0

	trues = []
	preds = []


	with open(filepath) as f:
		lines = f.readlines()
		for i in range(len(lines)):
			if(len(lines[i]) == 1):
				continue
			else:
				cnt = cnt +1
				txt = lines[i].split()
				word = txt[0]
				true = txt[1]
				pred = txt[2]
				print(word)
				print(len(true))
				print(len(pred))
				print(type(true))
				print(type(pred))
				trues.append(true)
				preds.append(pred)

				print('a' == 'a')
				print(Counter(true) == Counter(pred))
				if (true == pred):
					equal = equal + 1

	report_result = classification_report(trues, preds, output_dict=True)
	print(classification_report(trues, preds))
	print(report_result)


	print("total lenghth  : "),
	print(cnt)

	print("equal : "),
	print(equal)


	print("Accuracy : %4f" % report_result['accuracy'])
	print("Macro avg: %4f" % report_result['macro avg']['f1-score'])
	print("Weighted avg: %4f" % report_result['weighted avg']['f1-score'])

	return {'p':precision, 'r':recall, 'f1':f1score}


if __name__ == '__main__':
