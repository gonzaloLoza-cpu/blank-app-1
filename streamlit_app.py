import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras


from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, roc_curve, auc) 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight


# ==================== PAGE SETUP ====================

st.title("🎈 My new streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title("Project Stroke Analysis of Smokers")
st.write(
    "This app analyzes risk of stroke based on data from the Framingham Heart Study."
)


# ==================== LOAD AND PREPARE DATA ====================

dataset = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

st.subheader("Dataset Overview")
st.write(dataset.head())
st.write("We had 3 Periods worth of data due to overlap and bias when using all three we decided to use Period 1 only for our analysis.")
st.markdown("""
- Most active participants; 4,434 entries
- Highest incidence of stroke; 5.5%
- Least releavant missing data; only 4 relevant features with missing values
""")

period_1 = dataset.groupby('PERIOD').get_group(1)


# ==================== FEATURE SELECTION ====================

st.subheader("Deciding on Features to Use")
st.write("We decided to drop the following features due to irrelevance or just inaccuracy for stroke prediction:")

period_1_relev = period_1.drop(columns=["HDLC", "LDLC", "RANDID", "TIME",
    "TIMESTRK", "TIMEAP", "TIMEMI", "TIMEMIFC", "TIMECHD",
    "TIMECVD", "TIMEDTH", "TIMEHYP", "GLUCOSE"])

st.write("HDLC, LDLC, RANDID, TIME, TIMESTRK, TIMEAP, TIMEMI, TIMEMIFC, TIMECHD, TIMECVD, TIMEDTH, TIMEHYP, GLUCOSE")
st.write("The resulting dataset is as follows:")
st.write(period_1_relev.head())


# ==================== TRAIN-TEST SPLIT ====================

st.subheader("Train Test Split")
st.write("We performed an 80-20 train-test split on the dataset to prepare for modeling.")

y = period_1_relev.STROKE
x = period_1_relev.drop(columns=["STROKE"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ==================== DATA CLEANING ====================

st.subheader("Handling Missing Data")
st.write("We noticed that there were some missing values in the dataset. Here is a summary of the missing values:")

missing_summary = period_1_relev.isnull().sum()
st.write(missing_summary[missing_summary > 0])
st.write("These would have to be imputed however before we could impute, we have to remove outliers as to not skew our imputation.")

categorical_features = ['SEX', 'CURSMOKE', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'DIABETES', 'BPMEDS', 'DEATH', 'ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'CVD', 'HYPERTEN', 'educ']
numerical_features = ['AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'TOTCHOL']


# ==================== OUTLIER DETECTION AND REMOVAL ====================

st.subheader("Outlier Detection and Removal")
st.write("We used box plots to identify outliers in the numerical features.")

# Creates dropdown menue
BoxPlot_Raw = st.selectbox(
    "Select a raw variable to view:",
    numerical_features
    )

# Creates graph based on the chosen variable
plt.figure(figsize=(8, 4))
sns.boxplot(x=period_1_relev[BoxPlot_Raw])
plt.title(f'Box plot of {BoxPlot_Raw}')
st.pyplot(plt)
plt.clf()



st.write("We tried several methods such as 2*IQR and Z-score but ultimately decided to use biological limits defined by us to cap our data. We used winsorization to impute these values with NaN for now and later re-impute them.")

x_train_processed = x_train.copy(deep=True)
x_test_processed = x_test.copy(deep=True)

winsorization_caps = {
    'TOTCHOL': 500,
    'DIABP': 125,
    'SYSBP': 220,
    'BMI': 50,
    'HEARTRTE': 200
}

for col, cap_val in winsorization_caps.items():
    if col in x_train_processed.columns:
        original_train_count = len(x_train_processed[x_train_processed[col] > cap_val])
        x_train_processed[col] = np.where(x_train_processed[col] > cap_val, np.nan, x_train_processed[col])

    if col in x_test_processed.columns:
        original_test_count = len(x_test_processed[x_test_processed[col] > cap_val])
        x_test_processed[col] = np.where(x_test_processed[col] > cap_val, np.nan, x_test_processed[col])

st.write("After applying winsorization, here is how our graphs look now:")

# Creates dropdown menue
BoxPlot_Winzorized = st.selectbox(
    "Select a winzorized variable to view:",
    numerical_features
    )

# Creates graph based on the chosen variable
plt.figure(figsize=(8, 4))
sns.boxplot(x=x_train_processed[BoxPlot_Winzorized])
plt.title(f'Box plot of {BoxPlot_Winzorized} after Winsorization')
st.pyplot(plt)
plt.clf()

st.write("Now that outliers have been handled, we can proceed to impute the missing values.")
# ==================== IMPUTING MISSING VALUES ====================

st.subheader("Imputing Missing Values")
st.write("Here is a heatmap showing the missing values before imputation:")

plt.figure(figsize=(10, 6))
sns.heatmap(x_train_processed.isnull(), cbar=False)
plt.title('Heatmap of missing values')
st.pyplot(plt)
plt.clf()

st.write("We used the most common imputation method to fill in the missing values for our categorical variables and KNN imputation for our numerical variables. When imputing we also made sure to:")
st.markdown("""
- Choose a appropriate method for each type of data
- Avoid data leakage by fitting imputers only on training data and then applying to test data
""")

missing_columns = [col for col in x_train_processed.columns if x_train_processed[col].isnull().any()]

def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

cat_miss = intersection(missing_columns, categorical_features)
num_miss = intersection(missing_columns, numerical_features)

x_train_final = x_train_processed.copy()
x_test_final = x_test_processed.copy()

# Categorical imputation
if len(cat_miss) > 0:
    cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    x_train_final[cat_miss] = cat_imputer.fit_transform(x_train_final[cat_miss])
    x_test_final[cat_miss] = cat_imputer.transform(x_test_final[cat_miss])

# Numerical imputation
knn_imputer = KNNImputer(n_neighbors=10, weights="uniform")
num_miss_train = [col for col in num_miss if col in x_train_final.columns]
num_miss_test = [col for col in num_miss if col in x_test_final.columns]

if len(num_miss_train) > 0:
    x_train_final[num_miss_train] = knn_imputer.fit_transform(x_train_final[num_miss_train])

if len(num_miss_test) > 0:
    x_test_final[num_miss_test] = knn_imputer.transform(x_test_final[num_miss_test])

st.write("After imputation, we double checked for any remaining missing values:")

plt.figure(figsize=(10, 6))
sns.heatmap(x_train_final.isnull(), cbar=False)
plt.title('Heatmap of missing values')
st.pyplot(plt)
plt.clf()

st.write("No missing values remain thus we could move on to programming the models")


# ==================== SCALING ====================

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_final)
x_test_scaled = scaler.transform(x_test_final)


# ==================== MODEL 1: INITIAL LOGISTIC REGRESSION ====================

st.write("Now onto the first modeling phase!")

model_initial = LogisticRegression(class_weight='balanced', max_iter=1000)
model_initial.fit(x_train_scaled, y_train)

y_pred = model_initial.predict(x_test_scaled)

st.subheader("Initial Model Results")
st.write("Here are the results from our initial Logistic Regression model:")
st.text(classification_report(y_test, y_pred))
st.write("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model_initial, x_test_scaled, y_test)
st.pyplot(plt)
plt.clf()
st.write("The initial model provides a baseline for our analysis. Next, we will explore feature selection techniques to improve model performance.")

# ==================== MODEL 3: RFE FEATURE SELECTION ====================

st.write("Now onto feature selection using Recursive Feature Elimination (RFE).")

estimator = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
n_features_to_select = 20

selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
selector.fit(x_train_final, y_train)

featureSupport = pd.DataFrame(data=selector.ranking_, index=list(x_train_processed.columns), columns=['Feature ranking'])

st.subheader("Feature Selection with Wrapper Method (RFE)")
st.write("We applied Recursive Feature Elimination (RFE) to select the most important features for our model. Here are the top ranked features:")

plt.figure(figsize=(10, 20))
sns.heatmap(featureSupport.sort_values(ascending=True, by='Feature ranking'), annot=True, cmap='viridis')
plt.title('Wrapper selection of features using RFE')
plt.ylabel('Features')
plt.xlabel('Ranking')
plt.tight_layout()
st.pyplot(plt)
plt.clf()

st.write("The RFE process helped us identify the most relevant features, which we can now use to refine our model further.")

selected_features_rfe = x_train_final.columns[selector.support_]
X_train_selected_rfe = x_train_final[selected_features_rfe]
X_test_selected_rfe = x_test_final[selected_features_rfe]

logreg_rfe = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
logreg_rfe.fit(X_train_selected_rfe, y_train)

y_pred_rfe = logreg_rfe.predict(X_test_selected_rfe)

st.subheader("Model Results with RFE Selected Features")
st.write("Here are the results from our Logistic Regression model using features selected by RFE:")
st.text(classification_report(y_test, y_pred_rfe))
st.write("Accuracy:", accuracy_score(y_test, y_pred_rfe))

ConfusionMatrixDisplay.from_estimator(logreg_rfe, X_test_selected_rfe, y_test)
plt.title('Confusion Matrix for RFE Selected Features')
st.pyplot(plt)
plt.clf()

st.write("Using RFE-selected features, our model maintained strong performance, demonstrating the effectiveness of feature selection in improving model efficiency without sacrificing accuracy.")


# ==================== MODEL 4: RANDOM FOREST CLASSIFIER ====================

st.subheader("Random Forest Classifier")

clf = RandomForestClassifier(max_depth=20, random_state=42)
clf.fit(x_train_final, y_train)

prediction = clf.predict(x_test_final)

st.write("Here are the results from our Random Forest Classifier model:")
st.text(classification_report(y_test, prediction))

ConfusionMatrixDisplay.from_predictions(y_true=y_test,
                                       y_pred=prediction,
                                       display_labels=clf.classes_)
st.pyplot(plt)
plt.clf()

st.write("The Random Forest Classifier provided a robust alternative to logistic regression, capturing complex patterns in the data and yielding competitive performance metrics.")

# ==================== PREPROCESSING PIPELINE ====================
st.subheader (" Preprocessing Pipeline Setup")
st.write("We set up a preprocessing pipeline to handle both numerical and categorical features appropriately to prevent data leakage when performing K-fold cross-validation" \
"   and hyperparameter tuning.")

# Numeric preprocessing 
numeric_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=10)),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="passthrough"
)



# ==================== CROSS-VALIDATED FOR L1 LOGISTIC REGRESSION ====================
st.subheader(" L1 Logistic Regression and Cross Validation") 
logreg_l1_pipe = Pipeline([
    ("prepocess", preprocessor),
    ("logreg", LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        penalty='l1',
        solver='liblinear',
        random_state=42
    ))
])
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}
cv_value = 5
cv_results_l1 = cross_validate(
    logreg_l1_pipe,
    x_train_final,
    y_train,
    scoring=scoring_metrics,
    cv=cv_value,
    return_train_score=False)
# Summarize cross-validation results
results_summary_l1 = {}
for metric_name in scoring_metrics:
    scores_embedded = cv_results_l1[f'test_{metric_name}']
    results_summary_l1[metric_name] = {
        'Mean': np.mean(scores_embedded),
        'Standard Deviation': np.std(scores_embedded)
    }
st.write("Cross-validation results for L1 Logistic Regression:"
         )
results_df_embedded = pd.DataFrame.from_dict(results_summary_l1, orient='index')
st.write(results_df_embedded)

logreg_l1_pipe.fit(x_train_final, y_train)

# Predict on test
y_pred_l1_test = logreg_l1_pipe.predict(x_test_final)
y_proba_l1_test = logreg_l1_pipe.predict_proba(x_test_final)[:, 1]

print("\nL1 Logistic Regression – Test set metrics:")
print(classification_report(y_test, y_pred_l1_test, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_l1_test))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_l1_test)
plt.show()
st.write("L1 Logistic Regression – Test set metrics:")
st.text(classification_report(y_test, y_pred_l1_test, digits=3))
st.write("ROC-AUC:", roc_auc_score(y_test, y_proba_l1_test))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_l1_test)
st.pyplot(plt)
plt.clf()
# =================== Hypertuned Random Forest ====================
st.subheader("Hypertuned Random Forest Classifier")

st.write("Before Cross-Validating ")




best_params = {
    "rf__bootstrap": True,
    "rf__class_weight": "balanced",
    "rf__max_depth": 35,
    "rf__max_features": "sqrt",
    "rf__max_samples": 0.9858560476945519,
    "rf__min_samples_leaf": 3,
    "rf__min_samples_split": 2,
    "rf__n_estimators": 504
}

# RF pipeline and preprocessor

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
    ))
])
best_rf_pipeline = rf_pipeline.set_params(**best_params)

# Reproducible CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Same scoring setup you used
scoring_metrics = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1_class1": make_scorer(f1_score, pos_label=1, average="binary"),
    "precision_class1": make_scorer(precision_score, pos_label=1, average="binary", zero_division=0),
    "recall_class1": make_scorer(recall_score, pos_label=1, average="binary", zero_division=0),
}

print("Running 5-fold CV with fixed best RF parameters...")

cv_results = cross_validate(
    best_rf_pipeline,
    x_train_final,
    y_train,
    scoring=scoring_metrics,
    cv=cv,
    return_train_score=False,
    error_score="raise",
)

# Summarize mean/std
summary = {}
for m in scoring_metrics:
    scores = cv_results[f"test_{m}"]
    summary[m] = {"Mean": np.mean(scores), "Std": np.std(scores)}

results_df = pd.DataFrame(summary).T
print("\n=== Fixed Best RF params: 5-fold CV results ===")
print(results_df)

print("\nBest params used:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
st.write("Cross-validation results for Hypertuned Random Forest Classifier:"
         )
results_df = pd.DataFrame(summary).T
st.write(results_df)
st.write("Best params used:")
for k, v in best_params.items():
    st.write(f"  {k}: {v}")
# Fit on full training data 


# ---- Params ----
EPOCHS = 100
BATCH_SIZE = 16
RANDOM_STATE = 42

# ---- Ensure y is 1D ----
y_train_arr = np.asarray(y_train).ravel()
y_test_arr   = np.asarray(y_test).ravel()

# ---- Calculate class weights ----
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_arr),
    y=y_train_arr
)
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

# ---- Model builder: SciKeras passes meta with feature count ----
def create_mlp_model(meta):
    n_features = meta["n_features_in_"]

    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_features,),
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(64, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(32, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# ---- Pipeline: scaler + NN ----
nn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", KerasClassifier(
        model=create_mlp_model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=class_weights # Pass the pre-calculated weights directly
    ))
])

# ---- CV ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def custom_roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

scoring_metrics = {
    "accuracy": "accuracy",
    "roc_auc": make_scorer(custom_roc_auc),
    "f1_macro": make_scorer(f1_score, average="macro"),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
}

st.write("Running leakage-free 5-fold CV for MLP pipeline...")

cv_results = cross_validate(
    nn_pipe,
    x_train_final,
    y_train_arr,
    scoring=scoring_metrics,
    cv=cv,
    return_train_score=False,
    error_score="raise",
    n_jobs=1
)

summary = {}
for m in scoring_metrics:
    scores = cv_results[f"test_{m}"]
    summary[m] = {"Mean": float(np.mean(scores)), "Std": float(np.std(scores))}

cv_df = pd.DataFrame(summary).T
print("\nCV Results (mean ± std):")
print(cv_df.round(4))

print("\nTraining final MLP on full training set...")
final_nn = nn_pipe  
final_nn.fit(
    x_train_final,
    y_train_arr
)

# ---- Test once ----
print("\nEvaluating on test set once...")
y_test_proba = final_nn.predict_proba(x_test_final)[:, 1]
y_test_pred  = (y_test_proba >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(y_test_arr, y_test_pred, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_test_arr, y_test_pred))

print("Test ROC-AUC:", roc_auc_score(y_test_arr, y_test_proba))
st.subheader("Neural Network Classifier with Cross-Validation")
st.write("Cross-validation results for Neural Network Classifier:")
cv_df = pd.DataFrame(summary).T     
st.write(cv_df.round(4))
st.write("\nClassification report:")
st.text(classification_report(y_test_arr, y_test_pred, digits=4))
st.write("Test ROC-AUC:", roc_auc_score(y_test_arr, y_test_proba))
ConfusionMatrixDisplay.from_predictions(y_test_arr, y_test_pred)
st.pyplot(plt)
plt.clf()

st.write("The Neural Network Classifier demonstrated weaker performance, yielding worse results than our random forest classifier and logistic regression models. This suggests that for this particular dataset, simpler models may be more effective.")
st.subheader("Conclusion and evaluation of models")
st.write("In conclusion, we explored various modeling techniques to predict stroke risk using the Framingham Heart Study dataset. Our initial Logistic Regression model provided a solid baseline, which we improved upon using Recursive Feature Elimination (RFE) for feature selection. The Random Forest Classifier emerged as the best-performing model, effectively capturing complex patterns in the data and yielding strong performance metrics. Conversely, the Neural Network Classifier underperformed compared to simpler models, indicating that more complex architectures do not always guarantee better results. Overall, our analysis highlights the importance of model selection and feature engineering in predictive modeling tasks.")
st.write("A summary of the F1-scores for each model is as follows:")
f1_scores = {
    "Initial Logistic Regression": f1_score(y_test, y_pred),
    "RFE Logistic Regression": f1_score(y_test, y_pred_rfe),
    "Random Forest Classifier": f1_score(y_test, prediction),
    "Neural Network Classifier": f1_score(y_test_arr, y_test_pred)
}
f1_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1 Score'])
st.write(f1_df)
st.write("The Random Forest Classifier achieved the highest F1-score, indicating its superior ability to balance precision and recall in predicting stroke risk within this dataset.")
