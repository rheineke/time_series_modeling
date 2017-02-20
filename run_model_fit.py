"""Fit a variety of models to the data"""
import datetime as dt
import math
import pickle
import time
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, \
    RANSACRegressor, RidgeCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


import data
import utils

_PCA_KWARGS = dict(n_components=3)
_RANDOM_STATE = None


def read_train_test_frames():
    df = data.read_csv()
    # Dependent and independent features
    y_df = df[data.DEPENDENT]
    x_df = df[data.INDEPENDENTS]
    # Split into training and testing sets
    test_size = .20  # 10e6 observations on rank 3 justifies reduced test size
    split_kwargs = dict(test_size=test_size, random_state=_RANDOM_STATE)
    return train_test_split(x_df, y_df, **split_kwargs)


def scaled_pipelines():
    # Model parameters
    # RANSAC parameters
    # 500 max trials takes 90s
    ransac_kwargs = {
        'max_trials': 1000,
        'min_samples': 5000,
        'loss': 'absolute_loss',
        'residual_threshold': 2.0,
        'random_state': _RANDOM_STATE,
    }
    # Ridge CV parameters
    alphas = [.01, .1, 1, 10]
    # Model instances
    model_steps = [
        # LinearRegression(),
        # [PolynomialFeatures(degree=2), LinearRegression()],
        # [PolynomialFeatures(degree=3), LinearRegression()],
        # RANSACRegressor(base_estimator=LinearRegression(), **ransac_kwargs),
        # RANSACRegressor with polynomial regression?
        # RidgeCV(alphas=alphas),
        # LassoCV(),  # Alphas set automatically by default
        # ElasticNetCV(l1_ratio=0.5),  # Same as default
        # [PolynomialFeatures(degree=2), ElasticNetCV(l1_ratio=0.5)],
        # SGDRegressor(),
    ]
    # Pipelines
    pipelines = []
    for m in model_steps:
        # Steps
        common_steps = [
            StandardScaler(),
            PCA(**_PCA_KWARGS)
        ]
        model_steps = m if isinstance(m, list) else [m]
        steps = common_steps + model_steps
        pipelines.append(make_pipeline(*steps))
    return pipelines


def sample_pipelines(pca_kernels=None, svr_kernels=None):
    """
    Pipelines that can't be fit in a reasonable amount of time on the whole
    dataset
    """
    # Model instances
    model_steps = []
    if pca_kernels is None:
        pca_kernels = ['poly', 'rbf', 'sigmoid', 'cosine']
    for pca_kernel in pca_kernels:
        model_steps.append([
            KernelPCA(n_components=2, kernel=pca_kernel),
            LinearRegression(),
        ])
    if svr_kernels is None:
        svr_kernels = ['poly', 'rbf', 'sigmoid']
    for svr_kernel in svr_kernels:
        model_steps.append(SVR(kernel=svr_kernel, verbose=True, cache_size=1000))
    # Pipelines
    pipelines = []
    for m in model_steps:
        # Steps
        common_steps = [
            StandardScaler(),
        ]
        model_steps = m if isinstance(m, list) else [m]
        steps = common_steps + model_steps
        pipelines.append(make_pipeline(*steps))
    return pipelines


def unscaled_pipelines():
    # Random forest parameters
    random_forest_kwargs = {
        'n_estimators': 10,
        'criterion': 'mse',
        'random_state': _RANDOM_STATE,
        'n_jobs': cpu_count(),
        'verbose': True,
    }
    # Gradient boosting parameters
    gradient_boost_kwargs = {
        'random_state': _RANDOM_STATE,
        'verbose': 1,
    }
    models = [
        DecisionTreeRegressor(max_depth=3, random_state=_RANDOM_STATE),
        # RandomForestRegressor(**random_forest_kwargs),
        # GradientBoostingRegressor(**gradient_boost_kwargs),
    ]
    pipelines = []
    for m in models:
        # Steps
        common_steps = [StandardScaler(), PCA(**_PCA_KWARGS)]
        steps = common_steps + [m]
        pipelines.append(make_pipeline(*steps))
    # Completely unscaled pipelines
    for m in models:
        pipelines.append(make_pipeline(m))
    return pipelines


def fit_evaluate(X_train, X_test, y_train, y_test, pipeline):
    pipeline_nm = utils.pipeline_name(pipeline)
    print(pipeline_nm)

    # Fit model
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    end_time = time.perf_counter()
    print('Time elapsed to fit: {:.1f}s'.format(end_time - start_time))

    # Evaluate model
    start_time = time.perf_counter()
    utils.evaluate(X_train, X_test, y_train, y_test, pipeline)
    end_time = time.perf_counter()
    print('Time elapsed to evaluate: {:.1f}s'.format(end_time - start_time))

    train_exponent = int(math.log10(len(X_train)))
    train_sample_n = int(math.pow(10, max(train_exponent - 2, 2)))
    train_sample_n = min(train_sample_n, 5000)
    X_sample_train = X_train.sample(n=train_sample_n)
    y_sample_train = y_train.reindex(X_sample_train.index)

    test_exponent = int(math.log10(len(X_test)))
    test_sample_n = int(math.pow(10, max(test_exponent - 2, 2)))
    test_sample_n = min(test_sample_n, 2000)
    X_sample_test = X_test.sample(n=test_sample_n)
    y_sample_test = y_test.reindex(X_sample_test.index)

    # Visually inspect residuals for goodness of fitness
    res_fig = utils.plot_residuals(X_sample_train,
                                   X_sample_test,
                                   y_sample_train,
                                   y_sample_test,
                                   pipeline)
    res_fmt = 'output/residual_{}.png'
    res_fig.savefig(res_fmt.format(pipeline_nm), dpi=200)

    # Learning curve
    # start_time = time.perf_counter()
    # learn_fig = utils.plot_learning_curve([pipeline], X_sample_train, y_sample_train)
    # lc_fmt = 'output/learning_curve_{}.png'
    # learn_fig.savefig(lc_fmt.format(pipeline_nm), dpi=200)
    # end_time = time.perf_counter()
    # print('Time elapsed for learning curves: {:.1f}s'.format(end_time - start_time))

    # Validation curve
    # start_time = time.perf_counter()
    # val_fig = utils.plot_validation_curve([pipeline],
    #                                       X_sample_train,
    #                                       y_sample_train,
    #                                       n_jobs=1)
    # vc_fmt = 'output/validation_curve_{}.png'
    # val_fig.savefig(vc_fmt.format(pipeline_nm), dpi=200)
    # end_time = time.perf_counter()
    # print('Time elapsed for validation curves: {:.1f}s'.format(end_time - start_time))


def scaled_train_test_split(X_train, X_test, y_train, y_test):
    # Standardize y - the pipeline will scale x
    sc_y = StandardScaler()
    y_train_std = sc_y.fit_transform(y_train)
    # For consistency, create a dataframe version of the np array
    y_train_std_df = pd.Series(data=y_train_std, index=y_train.index)
    y_test_std = sc_y.transform(y_test)
    # For consistency, create a dataframe version of the np array
    y_test_std_df = pd.Series(data=y_test_std, index=y_test.index)
    return [X_train, X_test, y_train_std_df, y_test_std_df]


def sampled_train_test_split(X_train, X_test, y_train, y_test, n=1000):
    X_train_sample = X_train.sample(n=n)
    y_train_sample = y_train.reindex(X_train_sample.index)
    return [X_train_sample, X_test, y_train_sample, y_test]


def persist_pipelines(pipelines):
    Path('models').mkdir(exist_ok=True)
    fp_fmt = 'models/{}-{:%y-%m-%d}.pkl'
    now = dt.datetime.now()
    for pipe in pipelines:
        print(utils.pipeline_name(pipe))
        fp_name = fp_fmt.format(utils.pipeline_name(pipe), now)
        joblib.dump(pipe, fp_name)
        # Pickle fails to work on RandomForestRegressor
        # with open(fp_name, 'wb') as fp:
        #     pickle.dump(pipe, fp)


if __name__ == '__main__':
    train_test_args = read_train_test_frames()
    scaled_train_test_args = scaled_train_test_split(*train_test_args)

    # Generate a list of pipelines, one for each model to be fit on scaled data
    scaled_pipes = scaled_pipelines()
    unscaled_pipes = unscaled_pipelines()
    pipes = scaled_pipes + unscaled_pipes

    # Print a summary so we have an idea of how many models are being run
    print('Number of models fit to entire data set: {}'.format(len(pipes)))

    for scaled_pipe in pipes:
        fit_evaluate(*scaled_train_test_args, scaled_pipe)

    sample_train_test_args = sampled_train_test_split(*scaled_train_test_args,
                                                      n=10000)
    sample_pipes = sample_pipelines(pca_kernels=[], svr_kernels=[])
    for sample_pipe in sample_pipes:
        fit_evaluate(*sample_train_test_args, sample_pipe)

    # Persist models
    persist_pipelines(pipes)

    if _PCA_KWARGS['n_components'] == 1:
        X_train = scaled_train_test_args[0]
        y_train_df = scaled_train_test_args[2]
        # Display multiple scaled models against single PCA component by slicing
        # pipelines
        scaled_models = []
        nsteps_common = 2
        for pipe in pipes:
            nsteps = len(pipe.steps) - nsteps_common
            scaled_models.append(Pipeline(pipe.steps[-nsteps:]))
        # Transform X now by common steps
        scale_transform_pipe = Pipeline(pipes[0].steps[:nsteps_common])
        X_train_scale = scale_transform_pipe.transform(X_train)
        X_train_scale_df = pd.DataFrame(data=X_train_scale, index=X_train.index)
        utils.single_feature(X_train_scale_df, y_train_df, scaled_models)
        plt.show()
