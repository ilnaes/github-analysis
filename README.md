# Github analysis

Various analysis on the top 8000 (4000 for R) project descriptions on Github of various languages.

We also train various models to predict languages based purely on the project description.  Models include knn and LASSO in R and a Pytorch transformer (distilroberta) model.  Overall, the transformer model performed best on the holdout set (~67% vs ~59% from LASSO being next closest).

The relevant files are

* R/get_data.R - get data from Github API
* R/simple_analysis.Rmd ([html notebook](https://ilnaes.github.io/gh-analysis/simple_analysis.nb.html)) - EDA of repo information not including the description
* R/description_analysis.Rmd ([html notebook](https://ilnaes.github.io/gh-analysis/description_analysis.nb.html))- EDA of repo description
* R/model.Rmd ([html notebook](https://ilnaes.github.io/gh-analysis/model.html)) - Fitting LASSO and knn models
* app.py - Streamlit app
* src/\* - package for training Pytorch transformer
* train.ipynb - simply Jupyter notebook displaying transformer training and training curves

You can train the transformer model by running from the base directory

```sh
python -m src.train
```

After it trains, it will save the model weights in output and you can then run the Streamlit app to test out the model using

```sh
streamlit run app.py
```

To demo the app, visit the [Heroku app](https://ilnaes-gh-analysis.herokuapp.com/).  (Possibly let it boot up for a few minutes)
