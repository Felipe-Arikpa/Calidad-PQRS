from calidad_pqrs.models.preprocessing_process import preprocessing_process
from calidad_pqrs.models.tuning_process import tune_process_model
from calidad_pqrs.utils import load_model, build_predictions_dataframe, optimize_threshold, save_thresholds
from calidad_pqrs.config import MODEL_PROCESS_DIR



def train_process():

    data_procesada = preprocessing_process()
    X_test, y_test = tune_process_model(dataset=data_procesada, n_trials=66)
    tfidf_model = load_model(MODEL_PROCESS_DIR, 'salud_process_classifier.pkl')

    predictions_dataframe = build_predictions_dataframe(
        model = tfidf_model,
        X_test = X_test,
        true_class_col='clase_real',
        y_test = y_test,
         predicted_class_col='clase_predicha',
        dataset_original = data_procesada,
        drop_cols = ['Proceso', 'Causa']
    )

    thresholds = optimize_threshold(
        dataset = predictions_dataframe,
        threshold_error = 0.05
    )

    save_thresholds(thresholds, 'process_thresholds.parquet')



if __name__ == "__main__":

    train_process()