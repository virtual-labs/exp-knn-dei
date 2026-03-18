## Procedure

The objective of this experiment is to classify iris flowers into three categories - Setosa, Versicolor, and Virginica based on a set of morphological features. The input to the model consists of four independent variables representing sepal and petal measurements, while the output is a categorical dependent variable indicating the species of the flower. In this experiment, the k-Nearest Neighbours (KNN) algorithm is used to perform classification based on distance similarity. The effectiveness of the model is evaluated using standard performance metrics.

**Step 1:** Import required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

**Step 2:** Load the Iris dataset using `load_iris()` from `sklearn.datasets`. The dataset has 150 samples and 5 columns, where four columns are input features (X) and one column is the output label (Y).

**Step 3:** Extract the feature matrix X, target labels y, feature names, and target class names.

**Step 4:** Create a Pandas DataFrame from extracted feature values.

**Step 5:** Add class label names as a separate column in the DataFrame for interpretability.

**Step 6:** Perform exploratory data analysis using `head()`, `info()`, and `describe()`.

**Step 7:** Check class distribution with `value_counts()`.

**Step 8:** Check missing values using `isnull().sum()`.

**Step 9:** Visualize feature distributions using histograms with KDE for each class.

**Step 10:** Generate a correlation heatmap for numerical features.

**Step 11:** Use encoded target values and add them as a new DataFrame column.

**Step 12:** Define feature set X using four numerical attributes: sepal length, sepal width, petal length, and petal width.

**Step 13:** Define target variable y using encoded class labels.

**Step 14:** Split dataset into training and testing sets in 70%–30% ratio with stratified sampling and fixed random state.

**Step 15:** Apply feature scaling with `StandardScaler()` by fitting on training data and transforming both train and test sets.

**Step 16:** Binarize test labels using `label_binarize()` for multi-class ROC–AUC computation.

**Step 17:** Initialize the KNN classifier with `k = 10` neighbours.

**Step 18:** Train the KNN model on scaled training data.

**Step 19:** Predict class labels with `predict()` and class probabilities with `predict_proba()` on test data.

**Step 20:** Evaluate model performance using Accuracy, Precision (macro), Recall (macro), F1-Score (macro), and ROC–AUC (One-vs-Rest, multi-class).

**Step 21:** Generate classification report for all classes.

**Step 22:** Plot confusion matrix heatmap to visualize class-wise performance.