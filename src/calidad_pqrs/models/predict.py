from calidad_pqrs.utils import load_directory, load_data, mapping_data, define_service, define_f3, clean_text_TfIdf, load_model, build_predictions_dataframe, create_probability_col, load_thresholds, format_results, export_results
from calidad_pqrs.config import MODEL_PROCESS_DIR, MODEL_CAUSES_DIR, PROCESS_DICT, CAUSES_DICT



process_model = load_model(MODEL_PROCESS_DIR, 'salud_process_classifier.pkl')
process_thresholds = load_thresholds(filename = 'process_thresholds.parquet')

causes_model = load_model(MODEL_CAUSES_DIR, 'salud_causes_classifier.pkl')
causes_thresholds = load_thresholds(filename = 'causes_thresholds.parquet')



def prepare_data_for_prediction():

    paths = load_directory(directory='Predict')

    complaints = load_data(paths)
    complaints['RAC_process_raw'] = complaints['Proceso'].copy()
    complaints['RAC_causes_raw'] = complaints['Causa'].copy()

    complaints = mapping_data(complaints)

    complaints['Filtro 4_map'] = complaints.apply(define_service, axis=1)
    complaints['Filtro 3_map'] = complaints['Filtro 3'].apply(define_f3)

    complaints = clean_text_TfIdf(complaints)

    complaints.rename(
        columns={
            'Proceso': 'Temp_RAC_Process',
            'Causa': 'Temp_RAC_Causes'
        }, inplace=True
    )

    return complaints



def make_predictions(complaints):

    probability_matrix = build_predictions_dataframe(
        model = process_model,
        X_test = complaints['Descripción_TfIdf'],
        true_class_col = 'RAC_Process',
        y_test= complaints['Temp_RAC_Process'],
        predicted_class_col = 'Proceso',
        dataset_original = complaints,
        drop_cols = ['Temp_RAC_Process', 'grammatical_categories_removed', 'stopwords_removed', 'Filtro 3', 'Filtro 4']
    )

    probability_matrix = create_probability_col(dataset=probability_matrix, col_name='Process_Probability')

    results = build_predictions_dataframe(
        model = causes_model,
        X_test = probability_matrix[['Descripción_TfIdf', 'Proceso', 'Filtro 4_map', 'Filtro 3_map']],
        true_class_col = 'RAC_Causes',
        y_test= probability_matrix['Temp_RAC_Causes'],
        predicted_class_col = 'Causa_Sugerida',
        dataset_original = probability_matrix,
        drop_cols = ['Descripción_TfIdf', 'Filtro 4_map', 'Filtro 3_map']
    )

    results = create_probability_col(dataset=results, col_name='Causes_Probability')

    results.rename(columns={'Proceso': 'Proceso_Sugerido',}, inplace=True)

    return results
    


def process_validation(row):

    if (row['RAC_process_raw'] not in PROCESS_DICT.keys() and row['RAC_process_raw'] not in process_model.classes_):
        return 'Proceso desconocido'
    
    elif row['RAC_Process'] == row['Proceso_Sugerido']:
        return 'Proceso correcto'
    
    process = row['Proceso_Sugerido']
    prob = row['Process_Probability']

    if prob >= process_thresholds.get(process):
        return 'Proceso incorrecto'
    else:
        return 'Alta incertidumbre'
    


def causes_validation(row):

    if (row['RAC_causes_raw'] not in CAUSES_DICT.keys() and 
        row['RAC_causes_raw'] not in causes_model.classes_):
        return 'Causa desconocida'
    
    if row['RAC_Causes'] == row['Causa_Sugerida']:
        return 'Causa correcta'
    
    cause = row['Causa_Sugerida']
    prob = row['Causes_Probability']
    
    if prob >= causes_thresholds.get(cause):
        return 'Causa incorrecta'
    else:
        return 'Alta incertidumbre'



def final_validation(row):

    process_val = row['Validated_Process_Label']
    causes_val = row['Validated_Causes_Label']

    if (process_val == 'Proceso desconocido' or causes_val == 'Causa desconocida'):
         return 'Proceso o Causa desconocidos'
    
    if causes_val == 'Causa incorrecta':
        return 'Revisar'

    return 'No se identifican alertas'



def main():

    complaints = prepare_data_for_prediction()

    results = make_predictions(complaints=complaints)

    results['Validated_Process_Label'] = results.apply(process_validation, axis=1)
    results['Validated_Causes_Label'] = results.apply(causes_validation, axis=1)
    results['Final Validation'] = results.apply(final_validation, axis=1)

    export = format_results(results)

    export_results(export, 'Alertas calidad.xlsx')

    return results



if __name__ == "__main__":

    main()