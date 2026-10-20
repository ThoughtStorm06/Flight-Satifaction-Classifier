# Flight Satisfaction: Data Insights and Modeling Notes

This note summarizes the exploration and modeling currently in [basic-eda-and-model-fitting-using-catboost-rf-an.ipynb](basic-eda-and-model-fitting-using-catboost-rf-an.ipynb). The findings below are observations from this dataset, not causal conclusions.

## Data Insights

### Numeric features

- **Age:** Mean is about 39, median 40, and the range is 7-85. The distribution has peaks around ages 25 and 39, then drops sharply after 60 and especially after 70. Satisfaction varies non-linearly with age; the notebook notes higher satisfaction among many passengers aged 40-60 and lower satisfaction among younger and older groups. Tree models can capture this without binning; age bands are a possible alternative for simpler models.
- **Flight distance:** Mean is about 1,354 and median about 954, with a right-skewed, multi-peaked distribution. The IQR rule marks roughly 18,331 rows above its upper bound, but these appear to be plausible long-distance flights rather than data errors. Avoid removing them solely as outliers. A log transform or distance bands could help linear models; tree models generally do not need either.
- **Departure and arrival delays:** Both are highly right-skewed and concentrated at zero. Their correlation is about 0.84. Arrival delay has 227 missing values. Since the IQR is zero, an IQR outlier rule is not useful here. Keep extreme delays unless independently shown to be invalid. Imputation, delay bins, or a delayed/not-delayed flag are candidates for further testing.

### Service ratings and zero values

- The service-rating features are mostly ordinal ratings from 1 to 5, but some rows contain 0.
- About 94% of rows have no zero ratings. Zeros cluster particularly in wifi, online booking, online boarding, and departure/arrival convenience ratings.
- The three online ratings often have zero values together. Rows with multiple online zeros have much higher satisfaction rates in the notebook's segment checks; a zero may mean not applicable or not rated rather than the lowest rating.
- Treating zero as its own category or adding explicit zero-count flags is a candidate feature strategy. This has not been implemented in the current preprocessing class.

### Nominal features

- **Type of Travel** and **Class** show strong differences in satisfaction: business travel and Business class have higher rates than personal travel and the economy classes.
- **Customer Type** also differs: loyal customers are more often satisfied than disloyal customers.
- **Gender** has nearly equal satisfaction rates in the notebook's summary and appears to add little signal on its own.

## What the Notebook Uses

- Exploratory summaries and plots: missing-value checks, descriptive statistics, histograms, bar charts, correlations, group satisfaction rates, and segment comparisons.
- A stratified 80/20 train/holdout split with `random_state=42`.
- A custom preprocessing transformer that drops `id` and maps the four string features (`Gender`, `Customer Type`, `Type of Travel`, and `Class`) to integer values. The remaining categorical rating columns are cast to integers.
- Randomized hyperparameter search for Random Forest and LightGBM, with 4 parameter samples, 3-fold CV, and ROC-AUC scoring.
- A tuned MLP pipeline with median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for rating and nominal columns, and engineered zero-rating features.
- `RandomizedSearchCV` with 3-fold ROC-AUC scoring compares Random Forest, LightGBM, and MLP on the full training split. The tuned MLP led with **0.9567 mean CV ROC-AUC**, followed by Random Forest (**0.9540**) and LightGBM (**0.9468**). The chosen MLP parameters were `(64, 32)` hidden units, `alpha=1e-5`, and learning rate `0.001`.
- An RBF-kernel SVM is also tuned with 3-fold CV, but only on a stratified 8,000-row training sample because exact RBF SVM training scales poorly. It scored **0.9457 CV ROC-AUC** on that sample, so this is exploratory and not directly comparable to full-data CV results.
- On the untouched holdout split, ROC-AUC was **0.9561 for MLP**, **0.9531 for Random Forest**, **0.9471 for sampled RBF SVM** (on 20,000 holdout rows), and **0.9458 for LightGBM**. The MLP's full-training ROC-AUC was **0.9590**. Some MLP candidates still emitted convergence warnings at 60 iterations, so further training/tuning may change these results.
- The best full-data CV model is refit by the search object and used to generate the competition test probabilities. The submission is written to `kaggle/working/submission.csv`.

## What Is Not Yet Used or Complete

- Delay-derived features and age bands are still candidates for experimentation. Zero-rating flags/counts are used in the MLP pipeline, but not in the existing tree preprocessing transformer.
- The tree preprocessing transformer still maps nominal features to integers, while the MLP uses one-hot encoding. Compare these representations if tree-model tuning continues.
- The SVM result is based on a sample, not the full dataset. The current CV estimates use 3 folds and one random seed; repeated or nested validation would give a more robust estimate after model selection.

## Neural Network Note

A small feed-forward neural network (MLP) is a reasonable model for this tabular data, though its advantage should be confirmed with repeated validation rather than a single split.

1. Drop `id`. Fit all preprocessing on the training fold only to avoid leakage.
2. Impute missing numeric values and standardize continuous features such as age, distance, and delays. Neural networks are sensitive to feature scale.
3. One-hot encode nominal features. For the 1-5 service ratings, compare one-hot encoding with numeric ordinal encoding; preserve zero as its own category or add zero flags rather than silently treating it as the worst rating.
4. Use a modest dense network with ReLU hidden layers and a sigmoid output for the binary satisfaction probability. Train with binary cross-entropy, early stopping, and regularization such as dropout or weight decay.
5. Compare ROC-AUC and log loss with the individual tree models. Check the train/validation gap and probability calibration; do not choose a model based on training performance alone.

For a first pass, scikit-learn's `MLPClassifier` can provide a quick baseline when paired with a `ColumnTransformer` and a leakage-safe pipeline. A framework such as PyTorch or Keras gives more control if the baseline justifies further tuning.