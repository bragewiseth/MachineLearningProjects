from sklearn.linear_model import Lasso

model = Lasso(fit_intercept=False, alpha=0)
model.fit(X_train_scaled, y_train_scaled)
predict_lasso = model.predict(scaler.transform(poly.transform(X)[:,1:])) + y_train_mean # type: ignore
# predict_lasso = clf.predict(poly.transform(X)) 
y_tilde_train_lasso = model.predict(X_train_scaled) + y_train_mean
y_tilde_test_lasso = model.predict(X_test_scaled) + y_train_mean