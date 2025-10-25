from data.data_preprocessing_all import CSVDataLoader, DataPreprocessor
file_name = "C:/EC/BSE/DSDM/Term 1/21DM004 Computing for Data Science/hw5/process_data_project/process_data/data/sample_diabetes_mellitus_data.csv"

# Step 1: Load and split the data
loader = CSVDataLoader(file_name)
train_df, test_df = loader.split_data()

# Step 2: Preprocess the data
preprocessor = DataPreprocessor(train_df, test_df)
train_clean, test_clean = (
    preprocessor
    .remove_nans(['age', 'gender', 'ethnicity'])
    .fill_nans_with_mean(['height', 'weight'])
    .get_data()
)

# Step 3: Apply feature transformations
from features.feature_transform import BMICalculator, EthnicityEncoder, GenderBinaryEncoder
transformers = [
    BMICalculator(),
    EthnicityEncoder(),
    GenderBinaryEncoder()
]

# Transform training and test data
for transformer in transformers:
    train_clean = transformer.transform(train_clean)
    test_clean = transformer.transform(test_clean)

for transformer in transformers:
    print(f"- {transformer.__class__.__name__}: {transformer.get_feature_names()}")

print("Columns transformed: ",train_clean.columns)

# Step 4: Create and train the model
from model.predictor import DiabetesModel

# Define features and target
feature_cols = ['age', 'bmi', 'gender_M', 'gender_F'] + [col for col in train_clean.columns if col.startswith('ethnicity_')]
target_col = 'diabetes_mellitus'

# Initialize model with custom hyperparameters
model = DiabetesModel(
    feature_columns=feature_cols,
    target_column=target_col,
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42
    }
)

# Train the model
model.train(train_clean)

metrics = model.pred_auc_score(train_clean, test_clean)
print(metrics)

# Get probability predictions
prob_predictions = model.predict(test_clean)
print("\nProbability predictions shape:", prob_predictions.shape)
print("Sample probabilities (first 10 rows):")
print(prob_predictions[:10])