import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
df = pd.read_csv("submissions.csv",error_bad_lines=False)
df = df[df.username != None]
df = df[df.username.notnull()]
print(df.info())
tf = TfidfVectorizer()
item = set(df['#image_id'])
num  = df['#image_id'].value_counts(sort=False).tolist()
print(len(item) , len(num))
#print(len(item) , len(df['#image_id']))
c = 0
print(df['total_votes'][c:c+2])
cmt = []
train = []
for i in range(len(item)):
	ma = max(df['total_votes'][c:c+num[i]])
	for j in range(num[i]):
		if float(df['total_votes'][c+j:c+j+1]) == ma:
			#print(df[c+j:c+j+1])
			cmt.append(df[c+j:c+j+1])
			train.append(df['title'][c+j:c+j+1].str.lower().to_string())
			break
	c = c + num[i]
'''
print(cmt[0:1])
print(train[0:5])
print('---------------------')
tf_matrix = tf.fit_transform(train)
tf.get_feature_names()
tf_matrix.toarray()
print(tf_matrix[1:2])
print('---------------------')
kq = cosine_similarity(tf_matrix[1:2],tf_matrix)
print(max(kq[0]))
#cmt['kq'] = pd.Series(np.random.randn(kq[0]),index=cmt.index)
#print(type(cmt),cmt['total_votes'])
'''
print('-----------------------------')
nhap = '22198'
print(df.iloc[30])
temp = []
for i in range(len(train)):
	c = train[i].find(' ')
	temp.append(int(train[i][0:c]))
for i in range(len(df)):
	if i not in temp:
		df['username'][i:i+1]  = 0
df = df[df.username != 0]
tem = 0
for i in range(len(df)):
	if df['#image_id'].iloc[i] == nhap:
		tem = i
print(len(df),df.iloc[1])
tf_matrix = tf.fit_transform(df['title'])
#tf.get_feature_names()
tf_matrix.toarray()
kq = cosine_similarity(tf_matrix[0:4060],tf_matrix)   #tf_matrix[tem:tem+5]
df['point'] = kq[0]
print(df['point'])
print(df[['#image_id','title','point']].sort_values(by=['point'],ascending=False).head(15)) 
#,'username','point'
print('-----------------------------')
kq2 = cosine_similarity(tf_matrix[tem:tem+1],tf_matrix)   #
df['point2'] = kq2[0]
print(df['point2'])
print(df[['#image_id','title','point2']].sort_values(by=['point2'],ascending=False).head(15)) 
print('-----------------------------')
dk = df.iloc[0:5]
print(dk['title'])

tk = TfidfVectorizer()
tk_matrix = tk.fit_transform(dk['title'])
print(tk.get_feature_names())
tk_matrix.toarray()
print(tk_matrix.shape)
print(tk_matrix)
print(cosine_similarity(tk_matrix[0:1],tk_matrix))
