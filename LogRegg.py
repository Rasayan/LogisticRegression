import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasheet = pd.read_csv('Social_Network_Ads.csv');
x = datasheet.iloc[:, :-1].values;
y = datasheet.iloc[:, -1].values;

from sklearn.model_selection import train_test_split;
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# print(x);

from sklearn.preprocessing import StandardScaler;
sc = StandardScaler();
x_train[:,:] = sc.fit_transform(x_train[:,:]);
x_test = sc.fit_transform(x_test);

# print(x_train); 
 
from sklearn.linear_model import LogisticRegression;
classifier = LogisticRegression(random_state=0);
classifier.fit(x_train, y_train);

print(classifier.predict(sc.transform([[56,45000]])))

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))