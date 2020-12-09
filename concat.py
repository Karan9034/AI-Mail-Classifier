import os, csv
import pandas as pd

def Concat(test,pred):
	# combined_csv = pd.concat([pd.read_csv('pred.csv', encoding = "cp1252"), pd.read_csv('test.csv', encoding = "cp1252")])
	# combined_csv.to_csv( "combined_csv.csv", index=False, columns=['Subject','Date', 'Sender', 'Body', 'Body_Unformatted', 'Label'], encoding='cp1252')
	test = pd.read_csv(test, encoding='cp1252')
	pred = pd.read_csv(pred)
	combined_csv = pd.concat([test,pred], axis=1)
	combined_csv.to_csv(os.path.join(os.getcwd(),'test-uploads', 'model-output','result.csv', index=False)