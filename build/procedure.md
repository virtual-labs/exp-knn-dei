### Procedure

The objective of this experiment is to classify iris flowers into three categories - Setosa, Versicolor, and Virginica based on a set of morphological features. The input to the model consists of four independent variables representing sepal and petal measurements, while the output is a categorical dependent variable indicating the species of the flower. In this experiment, the k-Nearest Neighbours (KNN) algorithm is used to perform the classification based on distance similarity.

**Step 1:** Import the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

**Step 2:** Load the Iris dataset using `load_iris()` from `sklearn.datasets`. The dataset consists of 150 samples and 5 columns, where four columns correspond to input features (X) representing sepal and petal measurements, and one column corresponds to the output variable (Y) indicating the flower species.

**Step 3:** Extract the feature matrix X, target labels y, feature names, and target class names from the dataset.

**Step 4:** Create a Pandas DataFrame using the extracted feature values.

**Step 5:** Add the class label names as a separate column in the DataFrame to improve interpretability.

**Step 6:** Perform exploratory data analysis (EDA) using `head()`, `info()`, and `describe()` to understand the structure and summary statistics of the dataset.

**Step 7:** Check the distribution of classes using `value_counts()`.

**Step 8:** Check for missing values in the dataset using `isnull().sum()`.

**Step 9:** Visualize feature distributions using histograms with kernel density estimation (KDE) for each feature across different classes.

**Step 10:** Generate a correlation heatmap to analyse relationships among the numerical features.

**Step 11:** Encode the target labels using the already encoded target values provided by the dataset and add them as a new column in the DataFrame.

**Step 12:** Define the feature set X using the four numerical attributes: sepal length, sepal width, petal length, and petal width.

**Step 13:** Define the target variable y as the encoded class labels.

**Step 14:** Split the dataset into training and testing sets using a 70%–30% ratio with stratified sampling and a fixed random state to ensure reproducibility.

**Step 15:** Apply feature scaling using `StandardScaler()` by fitting the scaler on the training data and transforming both training and testing datasets to ensure fair distance computation.

**Step 16:** Binarize the test labels using `label_binarize()` to enable multi-class ROC–AUC computation.

**Step 17:** Initialize the k-Nearest Neighbours classifier with the number of neighbours set to k = 10.

**Step 18:** Train the KNN model using the scaled training dataset.

**Step 19:** Predict the class labels using `predict()` and class probabilities using `predict_proba()` on the test dataset.

**Step 20:** Evaluate the model performance using Accuracy, Precision (macro average), Recall (macro average), F1-Score (macro average), and ROC–AUC (One-vs-Rest, multi-class).

**Step 21:** Generate a classification report showing precision, recall, and F1-score for each class.

**Step 22:** Plot a confusion matrix heatmap to visualize the classification performance of the model.