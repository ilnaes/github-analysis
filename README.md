# Github analysis

Various analysis on the top 8000 (4000 for R) project descriptions on Github of various languages.

We also train various models to predict languages based purely on the project description.  Models include knn and LASSO in R and a transformer (distilroberta) model in Python.  Overall, the transformer model performed best on the holdout set (73% vs 67% from LASSO being next closest).

You can train the model by running from the base directory

```sh
python -m src.train
```

After it trains, it will save the model weights in output and you can then run the Streamlit app to test out the model using

```sh
streamlit run app.py
```

To demo the app, visit the [Heroku app](https://ilnaes-gh-analysis.herokuapp.com/).  (Possibly let it boot up for a few minutes)
