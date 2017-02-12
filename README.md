# README #

This project uses a large data set to demonstrate a variety of data science and
machine learning techniques

The [data set] for this machine learning tour is hosted on Google Drive. The 
file is a compressed text file containing comma-separated data in 10 columns. 
The goal is to build a model to predict the 10th column using the first 9
columns.

### Installation ###

* Install [Anaconda] or a similar Python environment. This project uses Python
3.x but 2.x may work.
* Install required packages: `pip install --user --upgrade -r requirements.txt`

### Data analysis ###

The project has 3 modules that can be run:
* Exploratory data analysis: `python run_eda.py`
* Fit a wide range of models to the data: `python run_model_fit.py`
* Fit a final model: `python run_model_fit.py`

[Anaconda]: https://www.continuum.io/downloads
[data set]: https://drive.google.com/open?id=0BwpUS-D5xBA9WnlMYU5sSkRHUmM