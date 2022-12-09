import seaborn as sns
import pickle
import numpy as np

df = sns.load_dataset('iris')
x = df.iloc[:,:-1]
print(x)
y = df.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x,y)

with open('rf.pkl','wb') as f:
    pickle.dump(model,f)

# with open('rf.pkl','rb') as f:
#     model = pickle.load(f)

# print(model.predict(np.array([2.4,2.5,0.5,0.5]).reshape(1,-1)))