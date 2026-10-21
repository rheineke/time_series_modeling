import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import run_model_fit
import utils


def frame(n=400, seed=0):
    """Columns 0-9 like train.txt: 0-8 mix three factors, 9 depends on them."""
    rng = np.random.RandomState(seed)
    factors = rng.randn(n, 3)
    df = pd.DataFrame(factors @ rng.randn(3, 10) + 0.05 * rng.randn(n, 10))
    df[data.DEPENDENT] = factors @ [3, -2, 1] + 0.1 * rng.randn(n)
    return df


def test_independents_drop_dependent():
    assert data.independents(frame()) == list(range(9))


def test_read_csv_names_columns(tmp_path):
    path = tmp_path / "train.txt.gz"
    frame(5).to_csv(path, header=False, index=False)
    df = data.read_csv(path)
    assert list(df.columns) == list(range(10))
    assert len(df) == 5


def test_read_csv_explains_missing_file(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        data.read_csv(tmp_path / "missing.txt.gz")
    assert "Please download the data set" in capsys.readouterr().out


def test_read_train_test_frames_holds_out_a_fifth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    frame(100).to_csv("input/train.txt.gz", header=False, index=False)
    X_train, X_test, y_train, y_test = run_model_fit.read_train_test_frames()
    assert (len(X_train), len(X_test)) == (80, 20)
    assert list(y_train.columns) == [data.DEPENDENT]
    assert 8 not in X_train.columns  # column 8 becomes the index


def test_pipeline_names():
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    assert utils.pipeline_name(pipe) == "Pipe_standardscaler_linearregression"
    assert utils.model_name(pipe) == "linearregression"


def test_pipeline_lists():
    assert len(run_model_fit.scaled_pipelines()) == 1
    assert len(run_model_fit.unscaled_pipelines()) == 1
    assert len(run_model_fit.sample_pipelines()) == 7
    assert len(run_model_fit.sample_pipelines([], ["rbf"])) == 1


def test_scaled_pipeline_fits_linear_signal(capsys):
    df = frame()
    X, y = df[data.independents(df)], df[[data.DEPENDENT]]
    (pipe,) = run_model_fit.scaled_pipelines()
    pipe.fit(X[:300], y[:300])
    utils.evaluate(X[:300], X[300:], y[:300], y[300:], pipe)
    assert "R^2 train" in capsys.readouterr().out
    assert pipe.score(X[300:], y[300:]) > 0.9


def test_scaled_split_standardizes_y():
    df = frame()
    X, y = df[data.independents(df)], df[[data.DEPENDENT]]
    _, _, y_train, y_test = run_model_fit.scaled_train_test_split(
        X[:300], X[300:], y[:300], y[300:]
    )
    assert y_train[0].mean() == pytest.approx(0, abs=1e-12)
    assert y_train[0].std(ddof=0) == pytest.approx(1)
    assert list(y_test.index) == list(range(300, 400))


def test_sampled_split_keeps_rows_aligned():
    df = frame()
    X, y = df[data.independents(df)], df[[data.DEPENDENT]]
    X_s, X_test, y_s, _ = run_model_fit.sampled_train_test_split(X, X, y, y, n=50)
    assert len(X_s) == 50 and len(X_test) == len(X)
    assert list(X_s.index) == list(y_s.index)


def test_plot_residuals_returns_figure():
    df = frame()
    X, y = df[data.independents(df)], df[data.DEPENDENT]
    model = LinearRegression().fit(X, y)
    fig = utils.plot_residuals(X[:300], X[300:], y[:300], y[300:], model)
    assert fig.axes[0].get_xlabel() == "Predicted values"
