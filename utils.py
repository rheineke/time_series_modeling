from sklearn.metrics import mean_squared_error, r2_score


def model_name(pipeline):
    return pipeline.steps[-1][0]


def evaluate(X_train, X_test, y_train, y_test, pipeline):
    # Visually inspect residuals for goodness of fitness
    # plot.residuals(X_train, X_test, y_train, y_test, pipeline)
    # plt.show()

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Mean squared error for the hell of it
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print('MSE train {:.3}, test {:.3}'.format(mse_train, mse_test))

    # Coefficient of determination
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print('R^2 train {:.3}, test {:.3}'.format(r2_train, r2_test))
