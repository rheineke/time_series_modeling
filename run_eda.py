"""Exploratory data analysis"""
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import andrews_curves, autocorrelation_plot, \
    lag_plot, parallel_coordinates, scatter_matrix
from sklearn.decomposition import PCA, KernelPCA

import data

if __name__ == '__main__':
    df = data.read_csv()

    # Description matrix
    print('Description:\n{}'.format(df.describe()))

    # Number of samples for visualization and other compute bound steps
    nsamples = 500

    # Scatterplot
    # Take a random sample of data rather than visualize all data
    sample_df = df.sample(nsamples)
    scatter_kwds = dict(alpha=0.2, figsize=(15, 15))

    diagonals = ['hist', 'kde']
    for d in diagonals:
        plt.clf()  # Clear any existing figure
        scatter_axes = scatter_matrix(sample_df, diagonal=d, **scatter_kwds)
        scatter_matrix_fp_fmt = 'output/scatter_matrix_{}.png'
        plt.savefig(scatter_matrix_fp_fmt.format(d))
    # Evaluation: Dimension 9 shows a curved relationship with dimension 0

    # Andrew's curves plot
    andrews_ax = andrews_curves(sample_df, data.DEPENDENT)
    andrews_ax.legend().set_visible(False)
    plt.savefig('output/andrews_curve.png')
    # Evaluation: Andrew's curves are for classification problem as far as I
    # know, but it makes an interesting plot regardless

    # Parallel coordinates plot
    parallel_ax = parallel_coordinates(sample_df, data.DEPENDENT)
    parallel_ax.legend().set_visible(False)
    plt.savefig('output/parallel_coordinates.png')
    # Evaluation: also not applicable here

    # Lag plot
    ncols = len(df.columns)
    figsize = figsize = (6.4 * ncols, 4.8)
    lag_fig, lag_axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    for i, c in enumerate(df.columns):
        lag_plot(sample_df[c], ax=lag_axes[i])
        plt.savefig('output/lag_plots.png')
    # Evaluation: lagged by one step, the results appear to be largely random

    # Autocorrelation plot
    ac_fig, autocorr_axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    for i, c in enumerate(df.columns):
        autocorrelation_plot(sample_df[c], ax=autocorr_axes[i])
        plt.savefig('output/autocorrelation_plots.png')
    # Evaluation: very low (<10%) autocorrelation among any of the variables

    # Correlation matrix
    corr_df = df.corr()
    print('Correlation matrix:\n{}'.format(corr_df))
    # Evaluation: 0, 2 and 6 have some degree of correlation with 9. Rest are
    # complete noise 0 and 2 have modest positive correlation. 6 is uncorrelated
    # to either. My expectation is that this is rank 3 with 2 large eigenvalues,
    # 1 small. Need to verify multicollinearity among these using PCA
    # All 3 independent variables have some degree of outliers and 0 has a
    # nonlinear relationship with 9 so try robust regression

    # PCA feature reduction
    # 99% of the variance can be explained by first PCA component, but the
    # relationship with 9 is still noisy. Try a polynomial regression
    pca = PCA(n_components=3)
    X = df[data.INDEPENDENTS]
    y = df[data.DEPENDENT]
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    # Evaluation: There is one eigenvector that explains almost all variance

    # PCA scatterplot
    nsamples_pca = nsamples * 10
    pca_df = pd.DataFrame(data=pca.fit_transform(X), index=df.index)
    pca_df[data.DEPENDENT] = y
    plt.clf()  # Clear any existing figure
    pca_sample_df = pca_df.sample(nsamples_pca)
    for d in diagonals:
        scatter_axes = scatter_matrix(pca_sample_df, diagonal=d, **scatter_kwds)
        plt.savefig(scatter_matrix_fp_fmt.format('pca_' + d))

    # Kernel PCA feature reduction
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    kernel_pcas = {}
    # Kernel PCAs are compute and memory intensive so fit on a random sample
    X_sample = X.sample(n=10000)
    for kernel in kernels:
        # pca = KernelPCA(kernel=kernel)
        # pca.fit(X_sample)
        # kernel_pcas[kernel] = pca
        print(kernel, dt.datetime.now())
