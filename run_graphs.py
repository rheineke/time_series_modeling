import datetime as dt

import pydotplus
from sklearn import tree
from sklearn.externals import joblib

if __name__ == '__main__':
    clf_name = 'decisiontreeregressor'
    model_date = dt.date.today()
    pipe_name = 'Pipe_{}'.format(clf_name)

    # Read pickled/joblib model
    fp_fmt = 'models/{}-{:%y-%m-%d}.pkl'
    tree_pipeline = joblib.load(fp_fmt.format(pipe_name, model_date))
    clf = tree_pipeline.named_steps[clf_name]

    # Create graphviz dot data, transform to pdf and write to file
    export_kwargs = dict(
        out_file=None,
        filled=True,
        rounded=True,
        special_characters=True
    )
    dot_data = tree.export_graphviz(clf, **export_kwargs)
    graph = pydotplus.graph_from_dot_data(dot_data)
    pdf_fmt = 'output/{}-{:%y-%m-%d}.pdf'
    graph.write_pdf(pdf_fmt.format(pipe_name, model_date))

    # From the command line, run:
    # dot -Tpdf output/Pipe_decisiontreeregressor.dot -o output/Pipe_decisiontreeregressor.pdf
