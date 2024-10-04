
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1], [2], [3], [4], [5],[9],[4],[8]])
y = np.array([1, 3, 2, 3, 5,2,3,4])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.scatter(X, y, color='blue', label='Actual Data')  
plt.plot(X_test, predictions, color='red', label='Predicted Line')  
plt.title('Linear Regression')  
plt.xlabel('X')  
plt.ylabel('y')  
plt.legend()  
plt.show()  
