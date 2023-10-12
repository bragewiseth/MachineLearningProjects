from numpy.core.fromnumeric import size
from numpy.lib import polynomial
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from utils import MSE, R2, Ridge,  plotFrankefunction, readData, printGrid
from sklearn.linear_model import Ridge as SkRidge






X, y  = readData("../data/syntheticData.csv")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


maxdegree = 5
numfeatures = int(((maxdegree+1)**2  + (maxdegree-1)) / 2) 
numlamdas = 50
lamdas = np.logspace(1,-6,numlamdas)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros((maxdegree,numlamdas))
testError  = np.zeros((maxdegree,numlamdas))
trainR2  = np.zeros((maxdegree,numlamdas))
testR2  = np.zeros((maxdegree,numlamdas))
betas = np.zeros((maxdegree,numlamdas,numfeatures))
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
scaler = StandardScaler()

for i, l in enumerate(lamdas):
    model = Ridge( alpha=l )
    for degree in range(maxdegree):
        poly = PolynomialFeatures(degree+1, include_bias=False)
        X_train = poly.fit_transform(x_train)
        X_test = poly.transform(x_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # type: ignore
        model.fit(X_train, y_train_scaled)
        betas[degree,i] = np.pad(model.beta, (0, numfeatures -model.beta.size))
        polydegree[degree] = degree + 1
        testError[degree,i] = MSE(y_test, model.predict(X_test) + y_train_mean)
        trainError[degree,i] = MSE(y_train, model.predict(X_train) + y_train_mean)
        testR2[degree,i] = R2(y_test, model.predict(X_test) + y_train_mean)
        trainR2[degree,i] = R2(y_train, model.predict(X_train) + y_train_mean)



skModel = SkRidge(fit_intercept=False, alpha= lamdas[numlamdas-1])
skModel.fit(X_train,y_train_scaled)

# compare ours with sklearnModel
print("sklearnModel beta: ", skModel.coef_)
print("our beta: ", betas[-1,-1])
print("sklearnModel MSE: ", MSE(y_test, skModel.predict(X_test) + y_train_mean))
print("our MSE: ", testError[-1,-1])
print("sklearnModel R2: ", R2(y_test, skModel.predict(X_test) + y_train_mean))
print("our R2: ", testR2[-1,-1])





printGrid(trainError, testError, trainR2, testR2, polydegree, lamdas)

poly = PolynomialFeatures(5, include_bias=False)
X = poly.fit_transform(X)
X = scaler.fit_transform(X)
print( 
    "confidence interval for beta when degree of polynomial = 5 lambda = 1.93: ",
    np.diag(np.var(y) * np.linalg.inv(X.T @ X + (1.93 * np.eye(X.shape[1]))) 
        @ X.T @ X @ (np.linalg.inv(X.T @X + 1.93 * np.eye(X.shape[1]))).T)
)



mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})

fig1 , ax1 = plt.subplots(figsize=(10,8))
ax1.xticks = lamdas
ax1.set_xscale("log")
ax1.set_ylabel(r"$\beta$",size=24)
ax1.set_xlabel(r"$\lambda$", size=24)
ax1.plot(lamdas, betas[4,:,:int(((polydegree[4]+1)**2  + (polydegree[4]-1)) / 2)])



plt.savefig("../runsAndAdditions/betaOverLambdaRidge5.png")


fig, ax = plt.subplots(figsize=(10,7))
ax.set_xticks(lamdas)
ax.set_xscale("log")
ax.plot(lamdas, trainR2[4], label="Train")
ax.plot(lamdas, testR2[4], label="Test")
ax.set_xlabel(r"$\lambda$",size=24)
ax.set_ylabel(r"$R^2$",size=24)
ax.legend()

plt.savefig("../runsAndAdditions/R2OverLambdaRidge5.png")









fig1 , ax1 = plt.subplots(5,1, figsize=(18,60))
fig1.suptitle(r"Ridge - $\mathbf{\beta}$ and model complexity for different $\lambda$")
fig1.text(0.5, 0.04, r'$\lambda$ for all $x$-axes', ha='center', va='center')
for i, ax in enumerate(ax1):
    ax.xticks = lamdas
    ax.set_xscale("log")
    ax.set_ylabel(r"$\beta$" ,size=24)
    ax.plot(lamdas, betas[i,:,:int(((polydegree[i]+1)**2  + (polydegree[i]-1)) / 2)], )
    ax.text(2, -0.15 + (0.2 * i), str(polydegree[i]) + '. order polynomial' , fontsize = 14, 
         bbox = dict(facecolor = 'red', alpha = 0.3))



plt.savefig("../runsAndAdditions/betaOverLambdaRidgeAll.png")


fig, ax = plt.subplots(5,1, figsize=(18,60))
fig.suptitle("Ridge - R2-score for different model complexity and $\lambda$")
fig.text(0.5, 0.04, r'Degree of polynomial for all $x$-axes', ha='center', va='center')
fig.text(0.06, 0.5, 'R2-score for all $y$-axes', ha='center', va='center', rotation='vertical')
for i,axi in enumerate(ax.flatten()):
    axi.set_xticks(lamdas)
    axi.set_xscale("log")
    axi.plot(lamdas, trainR2[i], label="Train")
    axi.plot(lamdas, testR2[i], label="Test")
    axi.legend()
    axi.text(2, 0.8, str(polydegree[i]) + '. order polynomial' , fontsize = 14, 
         bbox = dict(facecolor = 'red', alpha = 0.3))

plt.savefig("../runsAndAdditions/R2OverLambdaRidgeAll.png")




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
poly = PolynomialFeatures(5,include_bias=False)
z = model.predict(scaler.transform(poly.fit_transform(np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T ))) + y_train_mean
plotFrankefunction(xx,yy,z.reshape(100,100), (8,8), (1,1,1) , "Prediction using ridge")
plt.savefig("../runsAndAdditions/predictionRidge.png")

