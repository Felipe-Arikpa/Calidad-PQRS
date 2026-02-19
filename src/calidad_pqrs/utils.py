from pathlib import Path
import glob
import re
import json
import numpy as np
import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from calidad_pqrs.config import INPUT_DIR, INPUT_THRESHOLDS_DIR, OUTPUT_ALERTS_DIR, DICT_SERVICES_DIR, CAUSAS_ELIMINAR, PROCESOS_ELIMINAR, DESCONTENTO_PRODUCTO, CAUSES_DICT, PROCESS_DICT
from unidecode import unidecode
import pickle
from sklearn.metrics import f1_score
import warnings
# from sentence_transformers import SentenceTransformer, util



def load_directory(directory):

    directory = INPUT_DIR / directory

    return list(directory.glob("*"))



warnings.filterwarnings("ignore", message="Workbook contains no default style.*")

def load_data(data_location):

    print(f"\n{'═' * 80}")
    print('Leyendo las descripciones de las quejas...')

    data = pd.concat(
        [
            pd.read_excel(loc)[['Número del caso', 'Prestaciòn', 'Filtro 3', 'Filtro 4', 'Proceso', 'Causa', 'Descripción', 'Fecha de apertura']]\
                .dropna(subset=['Número del caso', 'Proceso', 'Causa', 'Descripción', 'Fecha de apertura'])
            for loc in data_location
        ],
        ignore_index=True
    )

    return data



def drop_data(dataset):

    df = dataset.copy()

    df = df[~df['Proceso'].isin(PROCESOS_ELIMINAR)]
    df = df[~df['Causa'].isin(CAUSAS_ELIMINAR)]

    return df



def mapping_data(dataset):

    print('Homologando procesos y causas con poca participación...')

    df = dataset.copy()

    Fix_DescontentoProducto_DescripcionPobre = df['Descripción'].isin(DESCONTENTO_PRODUCTO)

    Fix_ComunicacionPrestador_DirectorioMedico = (
        (df['Proceso'] == 'RELACIONAMIENTO Y SERVICIO AL CLIENTE') &
        (df['Causa'] == 'INCONFORMIDAD CON EL DIRECTORIO MEDICO')
    )

    Fix_SucursalVirtual_ComprarPagina = (
        (df['Proceso'] == 'ASESORIA Y VENTA') &
        (df['Causa'] == 'INSATISFACCION AL COMPRAR POR LA PAGINA')
    )

    Fix_filter4 = (df['Proceso'] != 'ASISTENCIA SALUD')
    Fix_Asistencia_filter4 = ((df['Proceso'] == 'ASISTENCIA SALUD') & (df['Filtro 4'].isna()))

    Fix_EvalyRelac_filter3 = ~df['Proceso'].isin(['EVALUACION', 'RELACIONAMIENTO Y SERVICIO AL CLIENTE', 'ASESORIA Y VENTA'])

    df.loc[Fix_DescontentoProducto_DescripcionPobre, ['Proceso', 'Causa']] = ('ASESORIA Y VENTA', 'DESCONTENTO CON EL PRODUCTO')
    df.loc[Fix_ComunicacionPrestador_DirectorioMedico, ['Proceso', 'Causa']] = ('ASISTENCIA SALUD', 'INCONFORMIDAD EN LA COMUNICACION CON EL PRESTADOR')
    df.loc[Fix_SucursalVirtual_ComprarPagina, ['Proceso', 'Causa']] = ('TRANSFORMACION DIGITAL', 'INCONVENIENTES GENERALES - SUCURSAL VIRTUAL')
    df.loc[Fix_filter4, ['Filtro 4']] = np.nan
    df.loc[Fix_Asistencia_filter4, ['Filtro 4']] = 'No identificado'
    df.loc[Fix_EvalyRelac_filter3, ['Filtro 3']] = 'Otros'


    df['Proceso'] = df['Proceso'].replace(PROCESS_DICT)
    df['Causa'] = df['Causa'].replace(CAUSES_DICT)

    return df



with open(DICT_SERVICES_DIR / 'dict.json', 'r', encoding='utf-8') as file:
    services_mapped = json.load(file)

def define_service(row):

    m_classif = row['Proceso']
    
    if m_classif != 'ASISTENCIA SALUD':
        return np.nan
    
    f4 = row['Filtro 4']
    causa = row['Causa']
    
    if pd.notna(f4):
        return services_mapped.get(f4.lower(), 'TBD')
    
    if pd.notna(causa):
        causa_upper = causa.upper()
        
        if 'INCAPACIDAD' in causa_upper:
            return 'Orden'
        elif 'MEDICAMENTO' in causa_upper:
            return 'Medicamentos'
        elif 'RED MEDICA' in causa_upper:
            return 'Consultas'
    
    return 'TBD'



def define_f3(valor):

    if pd.isna(valor):
        return np.nan
    
    valor_str = str(valor)
    
    if any(char.isdigit() for char in valor_str):
        return 'Ciudad'
    else:
        return 'Ente'
    


nlp = spacy.load('es_core_news_lg')

def remove_linguistic_features(txt):

    tokens = []
    eliminated = []

    doc = nlp(txt)

    for token in doc:

        if token.pos_ not in {'PRON', 'DET', 'NUM', 'ADP', 'CCONJ', 'SCONJ', 'AUX'}:
            tokens.append(token.text)
        else:
            eliminated.append(token.text)

    return ' '.join(tokens), ' '.join(eliminated)



def lemmatize_and_remove_stopwords(txt):

    tokens = []
    eliminated = []

    doc = nlp(txt)

    for token in doc:
        if token.lemma_ not in STOP_WORDS:
            tokens.append(token.lemma_)
        else:
            eliminated.append(token.text)

    return ' '.join(tokens), ' '.join(eliminated)



def clean_text_TfIdf(dataset):

    df = dataset.copy()

    df['Descripción_TfIdf'] = df['Descripción'].copy()

    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].str.lower()

    #mails
    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].apply(lambda txt: re.sub(r'\S+@\S+\.\S+', '', txt))

    #signos de puntuación
    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].apply(lambda txt: re.sub(r'[^\w\s]', '', txt, flags=re.UNICODE))

    #numeros
    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].apply(lambda txt: re.sub(r'\b\d+[\d\.,%-]*\b', '', txt))

    print('Removiendo categorías gramaticales que aportan poca información...')
    #Remover categorías gramaticales con poco aporte de información
    df['Descripción_TfIdf'], df['grammatical_categories_removed'] = zip(*df['Descripción_TfIdf'].apply(remove_linguistic_features))

    print('Lematizando palabras y removiendo stopwords...')
    #Lematizar y remover palabras vacías
    df['Descripción_TfIdf'], df['stopwords_removed'] = zip(*df['Descripción_TfIdf'].apply(lemmatize_and_remove_stopwords))

    #espacios multiples
    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].apply(lambda txt: re.sub(r'\s+', ' ', txt).strip())

    df['Descripción_TfIdf'] = df['Descripción_TfIdf'].apply(unidecode)

    return df



# def clear_text_Similarity(dataset):

#     df = dataset.copy()

#     df['Descripción_Similarity'] = df['Descripción'].copy()

#     df['Descripción_Similarity'] = df['Descripción_Similarity'].str.lower()

#     df['Descripción_Similarity'] = df['Descripción_Similarity'].str.replace("'", "", regex=False)

#     #espacios multiples
#     df['Descripción_Similarity'] = df['Descripción_Similarity'].apply(lambda txt: re.sub(r'\s+', ' ', txt).strip())

#     df = df.drop_duplicates(subset='Descripción_Similarity')

#     return df


# model_st = SentenceTransformer('intfloat/multilingual-e5-base')

# def calculate_similarities(text, embeddings_ref_dict, batch_size=32):

#     text_embeddings = model_st.encode(text,
#                                    convert_to_tensor=True,
#                                    normalize_embeddings=True,
#                                    batch_size=batch_size,
#                                    show_progress_bar=True)
    
#     results = {}
#     for process, ref_embeddings in embeddings_ref_dict.items():
#         similitudes = util.cos_sim(text_embeddings, ref_embeddings)
#         scores = similitudes.max(dim=1).values.cpu().numpy()
#         results[f'cos_sim_{process}'] = scores
    
#     return results



def save_model(model, model_dir, model_name, X_test, y_test):

    new_score = f1_score(y_test, model.predict(X_test), average='macro')
    
    model_path = model_dir / f'{model_name}'

    if model_path.exists():
        with model_path.open("rb") as file:
            saved_model = pickle.load(file)

        old_score = f1_score(y_test, saved_model.predict(X_test), average='macro')
    else:
        saved_model = None
        old_score = None

    if saved_model is None or new_score > old_score: # type: ignore
        with model_path.open("wb") as file:
            pickle.dump(model, file)

        print('\nNuevo modelo guardado.')
        print(f'Score nuevo modelo: {new_score}')
        print(f'Score nuevo anterior: {old_score}')
        print(f"\n{'═' * 80}")

    else:
        print('\nEl modelo guardado es mejor. No se actualiza.')
        print(f'Score modelo guardado: {old_score}')
        print(f'Score nuevo modelo: {new_score}')
        print(f"\n{'═' * 80}")



def build_predictions_dataframe(model, X_test, true_class_col, y_test, predicted_class_col, dataset_original, drop_cols):
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    resultados = pd.DataFrame({
        f'{true_class_col}': y_test,
        f'{predicted_class_col}': y_pred
    }, index=X_test.index)
    
    prob_cols = [f'prob_{clase}' for clase in model.classes_]
    proba_df = pd.DataFrame(y_proba, columns=prob_cols, index=X_test.index)
    
    final_df = (
        resultados
        .join(dataset_original, how='left')
        .join(proba_df, how='left')
        .drop(columns=drop_cols, errors='ignore')
    )
    
    return final_df



def load_model(model_dir, model_name):
    model_path = model_dir / f'{model_name}'

    with model_path.open("rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model



def optimize_threshold(dataset, threshold_error=0.12):
    
    results = []
    
    clases = dataset['clase_predicha'].unique()
    
    for clase in clases:
        
        df_filtered = dataset[dataset['clase_predicha'] == clase].copy()
        
        prob_col = f'{'prob_'}{clase}'
        
        true_positive = df_filtered[df_filtered['clase_real'] == clase]
        total_true_positive = len(true_positive)

        false_positive = df_filtered[df_filtered['clase_real'] != clase]
        total_false_positive = len(false_positive)
        
        thresholds = np.sort(df_filtered[prob_col].unique())[::-1]
        
        bets_threshold = None
        best_tp_rate = -1
        best_fp_rate = None
        
        for threshold in thresholds:
            
            fp_pass = np.sum(false_positive[prob_col] >= threshold)
            tp_pass = np.sum(true_positive[prob_col] >= threshold)
            
            fp_rate = fp_pass / total_false_positive if total_false_positive > 0 else 0
            tp_rate = tp_pass / total_true_positive if total_true_positive > 0 else 0
            
            if fp_rate <= threshold_error:
                
                if tp_rate > best_tp_rate:
                    best_tp_rate = tp_rate
                    best_fp_rate = fp_rate
                    bets_threshold = threshold
        
        if bets_threshold is not None:
            results.append({
                "Clase": clase,
                "Umbral": round(bets_threshold,4),
                "Falsos que pasan (%)": round(best_fp_rate, 2),
                "Verdaderos que pasan (%)": round(best_tp_rate, 2)
            })
        else:
            results.append({
                "Clase": clase,
                "Umbral": 0.8,
                "Falsos que pasan (%)": None,
                "Verdaderos que pasan (%)": None
            })
    
    return pd.DataFrame(results)



def save_thresholds(thresholds, filename):

    output_path = INPUT_THRESHOLDS_DIR / filename

    thresholds.to_parquet(output_path, index=False)



def load_thresholds(filename):

    thresholds = pd.read_parquet(INPUT_THRESHOLDS_DIR / filename)

    thresholds_dict = dict(zip(thresholds['Clase'], thresholds['Umbral']))

    return thresholds_dict



def create_probability_col(dataset, col_name):

    df = dataset.copy()

    df[col_name] = df.filter(like='prob_').max(axis=1)

    df.drop(columns=df.columns[df.columns.str.startswith('prob_')], inplace=True)

    return df



def format_results(dataset):

    df = dataset.copy()

    df.rename(
    columns={
        'RAC_process_raw': 'Proceso',
        'RAC_causes_raw': 'Causa',
    }, inplace=True
    )

    fix_empleado_sura = (
        (df['Causa_Sugerida'] == 'INADECUADA ATENCION DEL PROVEEDOR/PRESTADOR') &
        ~(df['Prestaciòn'].isna())
    )

    df.loc[fix_empleado_sura, ['Causa_Sugerida']] = 'INADECUADA ATENCION DEL EMPLEADO SURA'
    df.loc[df['Final Validation'] == 'No se identifican alertas', ['Proceso_Sugerido', 'Causa_Sugerida']] = (pd.NA, pd.NA)

    return df



def export_results(dataset, filename):

    export = dataset[[
        'Número del caso',
        'Descripción',
        'Proceso',
        'Causa',
        'Final Validation',
        'Proceso_Sugerido',
        'Causa_Sugerida',
        ]].copy()

    export.to_excel(OUTPUT_ALERTS_DIR / filename, index=False)

    print(f'\nResultados exportados en {OUTPUT_ALERTS_DIR / filename}')
    print(f"\n{'═' * 80}")