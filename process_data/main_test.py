from process_data import (
    CSVDataLoader, DataPreprocessor,
    BMICalculator, EthnicityEncoder, GenderBinaryEncoder,
    DiabetesModel,
)
from sklearn.metrics import roc_auc_score

file_name = "C:/EC/BSE/DSDM/Term 1/21DM004 Computing for Data Science/hw5/process_data_project/process_data/data/sample_diabetes_mellitus_data.csv"

# Load and split
loader = CSVDataLoader(file_name)
train_df, test_df = loader.split_data()

# Preprocess
preprocessor = DataPreprocessor(train_df, test_df)
train_clean, test_clean = (
    preprocessor
    .remove_nans(['age', 'gender', 'ethnicity'])
    .fill_nans_with_mean(['height', 'weight'])
    .get_data()
)

# Feature engineering
transformers = [BMICalculator(), EthnicityEncoder(), GenderBinaryEncoder()]
for t in transformers:
    train_clean = t.transform(train_clean)
    test_clean = t.transform(test_clean)

# Model
feature_cols = ['age', 'bmi', 'gender_M', 'gender_F'] + [
    c for c in train_clean.columns if c.startswith('ethnicity_')
]
target_col = 'diabetes_mellitus'

model = DiabetesModel(
    feature_columns=feature_cols,
    target_column=target_col,
    hyperparameters={'n_estimators': 200, 'max_depth': 10, 'random_state': 42},
)
model.train(train_clean)

# Predict + evaluate
probs = model.predict(test_clean)
pos_probs = probs[:, 1] if probs.ndim == 2 else probs
test_eval = test_clean.copy()
test_eval['predictions_prob'] = pos_probs
test_eval['predictions'] = (pos_probs >= 0.5).astype(int)

auc = roc_auc_score(test_eval[target_col], test_eval['predictions_prob'])
print(f"Test ROC AUC: {auc:.4f}")