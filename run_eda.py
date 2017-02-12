"""Exploratory data analysis"""
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import lag_plot, scatter_matrix
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
    scatter_kwargs = dict(alpha=0.2, figsize=(15, 15))

    diagonals = ['hist', 'kde']
    for d in diagonals:
        scatter_axes = scatter_matrix(sample_df, diagonal=d, **scatter_kwargs)
        scatter_matrix_fp_fmt = 'output/scatter_matrix_{}.png'
        plt.savefig(scatter_matrix_fp_fmt.format(d))

    # Lag plot
    plt.clf()  # Clear any existing figure
    lag_axes = lag_plot(sample_df)
    plt.savefig('output/lag_plot.png')

    # Correlation matrix
    corr_df = df.corr()
    print('Correlation matrix:\n{}'.format(corr_df))
    # 0, 2 and 6 have some degree of correlation with 9. Rest are complete noise
    # 0 and 2 have modest positive correlation. 6 is uncorrelated to either. My
    # expectation is that this is rank 3 with 2 large eigenvalues, 1 small.
    # Need to verify multicollinearity among these using PCA
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

    # PCA scatterplot
    nsamples_pca = nsamples * 10
    pca_df = pd.DataFrame(data=pca.fit_transform(X), index=df.index)
    pca_df[data.DEPENDENT] = y
    scatter_axes = scatter_matrix(pca_df.sample(nsamples_pca), **scatter_kwargs)
    plt.savefig(scatter_matrix_fp_fmt.format('pca'))

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
