from pathlib import Path
from datetime import datetime
import pandas as pd
from calidad_pqrs.models.predict import main
from sklearn.metrics import f1_score
from calidad_pqrs.config import OUTPUT_MONITORING_DIR



def complaints_for_eval_causes(dataset):

    df = dataset.copy()

    df = df[df['RAC_Process'] == df['Proceso_Sugerido']]

    return df



def build_log():

    predictions = main()
    predictions_for_causes = complaints_for_eval_causes(dataset=predictions)

    LOG = {
        'current_date': datetime.now(),
        'date_min': predictions['Fecha de apertura'].min(),
        'date_max': predictions['Fecha de apertura'].max(),
        'complaints_number': len(predictions),                                                                                            # Con este valor evalúo el proceso
        'process_score': f1_score(predictions['RAC_Process'], predictions['Proceso_Sugerido'], average='macro'),
        'complaints_number_for_causes': len(predictions_for_causes),                                                                      # Con este valor evalúo las causas
        'causes_score': (f1_score(predictions_for_causes['RAC_Causes'], predictions_for_causes['Causa_Sugerida'], average='macro') if len(predictions_for_causes) > 0 else 0)
    }

    return LOG



def save_monitoring(monitoring_dir, data_dict, file_name):

    monitoring_path = monitoring_dir / f'{file_name}'
    new_row = pd.DataFrame([data_dict])

    if monitoring_path.exists():

        existing = pd.read_parquet(monitoring_path)
        updated = pd.concat([existing, new_row], ignore_index=True)

    else:
        updated = new_row

    updated.to_parquet(monitoring_path, index=False)

    return updated



def evaluate_log(dataset, umbral_process, umbral_causes):

    df = dataset[dataset['complaints_number'] > 90]
    df = df.tail(3)

    if len(df) < 3:
        return None

    avg_score_process = df['process_score'].mean()
    avg_score_causes = df['causes_score'].mean()

    if (avg_score_process < umbral_process) or (avg_score_causes < umbral_causes):

        print(f"\n{'🚨' * 66}")
        print('El modelo necesita reentrenarse. \nPor favor siga los pasos para el reentrenamiento especificados en el archivo README.md o contacte a Felipe Aricapa.')
        print(f"\n{'🚨' * 66}")

        return 'warning'
    
    return 'no warning'



def generate_logs():

    LOG = build_log()

    df_logs = save_monitoring(
        monitoring_dir=OUTPUT_MONITORING_DIR,
        data_dict=LOG,
        file_name='transactions_log.parquet'
    )

    evaluate_log(
        dataset=df_logs,
        umbral_process=0.4,
        umbral_causes=0.5
    )



if __name__ == "__main__":

    generate_logs()