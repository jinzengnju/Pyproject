from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
np.set_printoptions(suppress=True)
def init_data():
    path=os.path.abspath('..')
    data=np.loadtxt(os.path.join(path,'data/Feature30/data_csv'),delimiter=',')
    return data
def Linear():
    data = init_data()
    X = data[:, 1:]
    Y = data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_test_predict=model.predict(X_test)
    print(model.coef_.__len__())
    print(model.intercept_)
    print(mean_squared_error(y_test,y_test_predict))
def NonlinearRegression():
    data=init_data()
    X=data[:,1:]
    Y=data[:,0]
    polynomial_featurizer = PolynomialFeatures(degree=2)
    X_ploy=polynomial_featurizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_ploy, Y, test_size=0.3, random_state=42)

    print(polynomial_featurizer.get_feature_names())
    model_polynomial = LinearRegression()
    model_polynomial.fit(X_train, y_train)
    print('2 r-squared', model_polynomial.score(X_test, y_test))


if __name__=="__main__":
    NonlinearRegression()
    #Linear()