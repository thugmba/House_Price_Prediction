import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#
# 1. Input
#
USAhousing = pd.read_csv('USA_Housing.csv')

USAhousing.head()
USAhousing.info()

USAhousing.describe()

sns.heatmap(USAhousing.iloc[:,0:6].corr())
plt.show()

plt.scatter(USAhousing['Avg. Area Income'],USAhousing['Price'])
plt.show()

plt.scatter(USAhousing['Avg. Area House Age'],USAhousing['Price'])
plt.show()

sns.pairplot(USAhousing)
plt.show()

sns.displot(USAhousing['Price'])

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#
# 2. Process
#
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()

#
# 3. Output
#
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
