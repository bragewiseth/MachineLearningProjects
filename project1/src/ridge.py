from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, makeData, Ridge, makeFigure, plotFrankefunction














np.random.seed(2023)
maxdegree = 5
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
n = 100
numlamdas = 50
lamdas = np.logspace(1,-6,numlamdas)
X, y, x_train, x_test, y_train, y_test = makeData(n, rand=0.1)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros((maxdegree,numlamdas))
testError  = np.zeros((maxdegree,numlamdas))
trainR2  = np.zeros((maxdegree,numlamdas))
testR2  = np.zeros((maxdegree,numlamdas))
betas = np.zeros((maxdegree,numlamdas,numfeatures))
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
scaler = StandardScaler()
model = Ridge()

for i, l in enumerate(lamdas):
    for degree in range(maxdegree):
        poly = PolynomialFeatures(degree+1, include_bias=False)
        X_train = poly.fit_transform(x_train)
        X_test = poly.transform(x_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # type: ignore
        model.fit(X_train, y_train_scaled, alpha=l)
        betas[degree,i] = np.pad(model.beta, (0, numfeatures -model.beta.size))
        polydegree[degree] = degree + 1
        testError[degree,i] = MSE(y_test, model.predict(X_test) + y_train_mean)
        trainError[degree,i] = MSE(y_train, model.predict(X_train) + y_train_mean)
        testR2[degree,i] = R2(y_test, model.predict(X_test) + y_train_mean)
        trainR2[degree,i] = R2(y_train, model.predict(X_train) + y_train_mean)





def printGrid(trainerror, testerror, trainR2, testR2, polydegree, lamdas):
    for i in range(len(polydegree)):
        testi  = np.argmin(testerror[i])    # find index of minimum test error (best fit)
        traini = np.argmin(trainerror[i])   # find index of minimum train error (best fit)
        testR2i  = np.argmax(testR2[i])     # find index of maximum test R2 (best fit)
        trainR2i = np.argmax(trainR2[i])    # find index of maximum train R2 (best fit)
        print("Degree of polynomial = ", polydegree[i])
        print("Best test error = {:25} \tfor λ = {}".format(testerror[i,testi],  lamdas[testi]))
        print("Best train error = {:24} \tfor λ = {}".format(trainerror[i,traini],  lamdas[traini]))
        print("Best test R2 = {:26} \tfor λ = {}".format(testR2[i,testR2i],  lamdas[testR2i]))
        print("Best train R2 = {:25} \tfor λ = {}".format(trainR2[i,trainR2i],  lamdas[trainR2i]))  

printGrid(trainError, testError, trainR2, testR2, polydegree, lamdas)





# mpl.rcParams.update({
#     'font.family': 'serif',
#     'mathtext.fontset': 'cm',
#     'font.size': '16',
#     'xtick.labelsize': '11',
#     'ytick.labelsize': '11',
#     # 'text.usetex': True,
#     'pgf.rcfonts': True,
# })
# fig1 , ax1 = plt.subplots(5,1)
# fig1.suptitle(r"$\mathbf{\beta}$ and model complexity for different $\lambda$")
# fig1.text(0.5, 0.04, r'$\lambda$ for all $x$-axes', ha='center', va='center')
# for i, ax in enumerate(ax1):
#     ax.xticks = lamdas
#     ax.set_xscale("log")
#     ax.set_ylabel(r"$\beta$")
#     ax.plot(lamdas, betas[i,:,:int(((polydegree[i]+1)**2  + (polydegree[i]-1)) / 2)])
#     ax.set_title(r"Degree of polynimoal" + " = " + str(polydegree[i]))



# plt.show()


# fig, ax = plt.subplots(5,1, figsize=(12,4))
# fig.suptitle("R2 for different model complexity and $\lambda$")
# fig.text(0.5, 0.04, r'Degree of polynomial for all $x$-axes', ha='center', va='center')
# fig.text(0.06, 0.5, 'R2-score for all $y$-axes', ha='center', va='center', rotation='vertical')
# for i,axi in enumerate(ax.flatten()):
#     # axi.set_xticks(lamdas)
#     axi.set_xscale("log")
#     axi.plot(lamdas, trainR2[i], label="Train")
#     axi.plot(lamdas, testR2[i], label="Test")
#     axi.legend()
#     axi.set_title(r"Degree of polynimoal" + " = " + str(polydegree[i]))

# plt.show()








# x = np.linspace(0,1,100)
# y = np.linspace(0,1,100)
# xx,yy = np.meshgrid(x,y)
# poly = PolynomialFeatures(5,include_bias=False)
# z = model.predict(scaler.transform(poly.fit_transform(np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T ))) + y_train_mean
# fig = makeFigure((8,8))
# plotFrankefunction(xx,yy,z.reshape(100,100), fig, (1,1,1) ,"Franke's Function")
# plt.show()