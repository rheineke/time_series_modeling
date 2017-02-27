import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve


# def classifier_name(classifier):
#     return str(type(classifier).__name__).lower()

def pipeline_name(pipeline):
    names = [nm for nm, _ in pipeline.steps]
    return 'Pipe_' + '_'.join(names)


def model_name(pipeline):
    return pipeline.steps[-1][0]
    # return classifier_name(pipeline.named_steps['clf'])


def evaluate(X_train, X_test, y_train, y_test, pipeline):
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Mean squared error for the hell of it
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print('MSE train {:.3}, validation {:.3}'.format(mse_train, mse_test))

    # Coefficient of determination
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print('R^2 train {:.3}, validation {:.3}'.format(r2_train, r2_test))


def plot_residuals(X_train_df, X_test_df, y_train_df, y_test_df, model):
    """Plot residuals of a random subset of train and test sets"""
    fig, axes = plt.subplots(nrows=1, ncols=1)

    # Random sample of training set
    # X_train_sample = X_train_df.sample(n=nsamples)
    # y_train_sample = y_train_df.reindex(X_train_sample.index)
    # Scatter plot of training residuals
    y_train_pred = model.predict(X_train_df)
    train_kwargs = dict(c='blue', marker='o', label='Training data')
    axes.scatter(y_train_pred, y_train_pred - y_train_df, **train_kwargs)
    # Random sample of test set
    # X_test_sample = X_test_df.sample(n=nsamples)
    # y_test_sample = y_test_df.reindex(X_test_sample.index)
    # Scatter plot of test residuals
    y_test_pred = model.predict(X_test_df)
    test_kwargs = dict(c='lightgreen', marker='o', label='Test data')
    axes.scatter(y_test_pred, y_test_pred - y_test_df, **test_kwargs)
    axes.set_xlabel('Predicted values')
    axes.set_ylabel('Residuals')
    axes.legend(loc='upper left')
    # Find range
    xmin = math.floor(min(y_train_pred.min(), y_test_pred.min()))
    xmax = math.ceil(max(y_train_pred.max(), y_test_pred.max()))
    axes.hlines(y=0, xmin=xmin, xmax=xmax, lw=2, color='red')
    axes.set_xlim([xmin, xmax])

    return fig


def plot_classication_data(X, y, test_idx=None):
    fig, axes = plt.subplots(nrows=1, ncols=1)

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        axes.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            c=cmap(idx),
            marker=markers[idx],
            label=cl,
            edgecolors='black',
            alpha=0.8
        )

    # highlight test samples
    test_kwds = dict(
        c='',
        edgecolors='black',
        linewidths=1,
        label='test set'
    )
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        axes.scatter(X_test[:, 0], X_test[:, 1], **test_kwds)

    return fig, axes


def plot_classifiers_decision_regions(classifiers, X, y, test_idx=None):
    """
    Plot one or more classifiers in the same image
    :param X:
    :param y:
    :param classifiers:
    :param test_idx:
    :return:
    """
    figsize = (6.4 * len(classifiers), 4.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(classifiers), figsize=figsize)

    for ax, clf in zip(axes, classifiers):
        _plot_decision_regions(ax, X, y, clf, test_idx=test_idx)

    return fig, axes


def _plot_decision_regions(axes, X, y, classifier, test_idx=None,
                           resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    axes.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    axes.set_xlim(xx1.min(), xx1.max())
    axes.set_xlim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        axes.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            c=cmap(idx),
            marker=markers[idx],
            label=cl,
            edgecolors='black',
            alpha=0.8
        )

    # highlight test samples
    test_kwds = dict(
        c='',
        edgecolors='black',
        linewidths=1,
        label='test set'
    )
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        axes.scatter(X_test[:, 0], X_test[:, 1], **test_kwds)


def plot_learning_curve(estimators, X, y, cv=10, n_jobs=1):
    figsize = (6.4 * len(estimators), 4.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(estimators), figsize=figsize)

    if len(estimators) == 1:
        axes = [axes]

    for ax, estimator in zip(axes, estimators):
        kwargs = dict(
            estimator=estimator,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            # scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        train_sizes, train_scores, test_scores = learning_curve(**kwargs)
        xlabel = 'Number of training samples'
        _plot_curve(ax, train_sizes, train_scores, test_scores, xlabel)
        ax.set_title(pipeline_name(estimator))
        # ax.set_title(classifier_name(estimator.named_steps['clf']))
    return fig


def plot_validation_curve(estimators, X, y, cv=10, **kwargs):
    figsize = (6.4 * len(estimators), 4.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(estimators), figsize=figsize)
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    if len(estimators) == 1:
        axes = [axes]

    for ax, estimator in zip(axes, estimators):
        vc_kwargs = dict(
            estimator=estimator,
            X=X,
            y=y,
            param_name='clf__C',
            param_range=param_range,
            cv=cv,
        )
        train_scores, test_scores = validation_curve(**vc_kwargs, **kwargs)
        xlabel = 'Parameter C'
        _plot_curve(ax, param_range, train_scores, test_scores, xlabel, 'log')
        ax.set_title(pipeline_name(estimator))
        # ax.set_title(classifier_name(estimator.named_steps['clf']))

    # fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    return fig


def _plot_curve(axes, train_sizes, train_scores, test_scores, xlabel,
                xscale=None):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    lbl = 'training accuracy'
    train_kwds = dict(color='blue', marker='o', markersize=5, label=lbl)
    axes.plot(train_sizes, train_mean, **train_kwds)
    axes.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color='blue'
    )

    lbl = 'validation accuracy'
    tst_kwds = dict(
        color='green',
        linestyle='--',
        marker='s',
        markersize=5,
        label=lbl
    )
    axes.plot(train_sizes, test_mean, **tst_kwds)
    axes.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color='green'
    )
    axes.grid()
    if xscale is not None:
        axes.set_xscale(xscale)
    axes.set_xlabel(xlabel)
    axes.set_ylabel('Accuracy')
    axes.legend(loc='upper right')
    # Calculate ymin
    min_train = np.min(train_mean - train_std)
    min_test = np.min(test_mean - test_std)
    ymin = np.round(min(min_train, min_test) - .05, decimals=1)
    axes.set_ylim([ymin, 1.0])
