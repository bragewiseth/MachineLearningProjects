import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils import FrankeFunction, readData, MSE, R2, OLS, plotFrankefunction
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression






#maximum degree of the polynomial
maxdegree = 5
X, y  = readData("../data/syntheticData.csv")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Arrays to store results for different polynomial degrees
polydegree = np.zeros(maxdegree)
trainError  = np.zeros(maxdegree)
testError  = np.zeros(maxdegree)
trainR2  = np.zeros(maxdegree)
testR2  = np.zeros(maxdegree)
betas = np.zeros((maxdegree,20))

#initializing standard scaler, our OLS model, the sklearn OLS model
scaler = StandardScaler()
model = OLS()
sklearnModel = LinearRegression(fit_intercept=False)
y_train_mean = np.mean(y_train)

for degree in range(maxdegree):
    #Creating polynomial features for the given degree
    poly = PolynomialFeatures(degree+1,include_bias=False)
    X_train = poly.fit_transform(x_train)
    X_test = poly.transform(x_test)
    #Scale the features using the scaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # type: ignore

    y_train_mean = np.mean(y_train)

    #Center the target variable
    y_train_scaled = y_train - y_train_mean

    #Fitting the OLS model
    model.fit(X_train, y_train_scaled)

    #Padding the beta values to match the expected shape
    betas[degree] = np.pad(model.beta,(0,20-model.beta.size))

    #Storing relevant metrics for the current degree
    polydegree[degree] = degree + 1
    testError[degree] = MSE(y_test, model.predict(X_test) + y_train_mean)
    trainError[degree] = MSE(y_train, model.predict(X_train) + y_train_mean)
    testR2[degree] = R2(y_test, model.predict(X_test) + y_train_mean)
    trainR2[degree] = R2(y_train, model.predict(X_train)+ y_train_mean)
    #Fit the scikit-learn model for the last degree
    if (degree == maxdegree -1) :
        sklearnModel.fit(X_train, y_train_scaled )



# compare ours with sklearnModel
print("sklearnModel beta: ", sklearnModel.coef_)
print("our beta: ", betas[-1])
print("sklearnModel MSE: ", MSE(y_test, sklearnModel.predict(X_test) + y_train_mean))
print("our MSE: ", testError[-1])
print("sklearnModel R2: ", R2(y_test, sklearnModel.predict(X_test) + y_train_mean))
print("our R2: ", testR2[-1])

# our train and test testError
print("Best train MSE: ", min(trainError))
print("Best train R2: ", max(trainR2))
print("Best test MSE: ", min(testError))
print("Best test R2: ", max(testR2))


poly = PolynomialFeatures(5, include_bias=False)
X = poly.fit_transform(X)
X = scaler.fit_transform(X)
print("confidence interval for beta when degree of polynomial = 5: ", np.diag(np.var(y) * np.linalg.inv(X.T @ X)))

#plot configuration for matplotlib
mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '14',
    'ytick.labelsize': '14',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})

fix1 , ax1 = plt.subplots(figsize=(10,10))
ax1.set_title(r"OLS - $\mathbf{\beta}$ and model complexity")
ax1.set_xlabel("Degree")
ax1.set_ylabel(r"values of $\beta$")
ax1.set(xticks=polydegree)
ax1.plot(polydegree, betas)
plt.savefig("../runsAndAdditions/betaOverOrderOLS.png")

fig, ax = plt.subplots(1,2, figsize=(12,6))
fig.suptitle("OLS")
ax[0].set_xlabel("Degree")
ax[1].set_xlabel("Degree")
ax[1].set_ylabel("R2")
ax[0].set_ylabel("MSE")
ax[0].set_title("MSE as a function of complexity")
ax[0].set(xticks=polydegree)
ax[1].set(xticks=polydegree)
ax[0].plot(polydegree, testError, label="test")
ax[0].plot(polydegree, trainError, label="train")
ax[0].legend()
ax[1].set_title("R2 as a function of complexity")
ax[1].plot(polydegree, testR2, label="test")
ax[1].plot(polydegree, trainR2, label="train")
ax[1].legend()
plt.savefig("../runsAndAdditions/R2andMSEOLS.png")




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
poly = PolynomialFeatures(5,include_bias=False)
z = model.predict(scaler.transform(poly.fit_transform(np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T ))) + y_train_mean
plotFrankefunction(xx,yy,z.reshape(100,100), (8,8), (1,1,1) ,"Prediction using OLS")
plt.show()
