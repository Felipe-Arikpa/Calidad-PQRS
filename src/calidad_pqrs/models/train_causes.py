from calidad_pqrs.models.preprocessing_causes import preprocessing_causes
from calidad_pqrs.models.tuning_causes import tune_causes_model
from calidad_pqrs.utils import load_model, build_predictions_dataframe, optimize_threshold, save_thresholds
from calidad_pqrs.config import MODEL_CAUSES_DIR



def train_causes():

    data_procesada = preprocessing_causes()
    X_test, y_test = tune_causes_model(dataset=data_procesada, n_trials=66)
    tfidf_model = load_model(MODEL_CAUSES_DIR, 'salud_causes_classifier.pkl')

    predictions_dataframe = build_predictions_dataframe(
        model = tfidf_model,
        X_test = X_test,
        true_class_col = 'clase_real',
        y_test = y_test,
        predicted_class_col = 'clase_predicha',
        dataset_original = data_procesada,
        drop_cols = 'Causa'
    )

    thresholds = optimize_threshold(
        dataset = predictions_dataframe,
        threshold_error = 0.05
    )

    save_thresholds(thresholds, 'causes_thresholds.parquet')



if __name__ == "__main__":

    train_causes()