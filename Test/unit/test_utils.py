import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from calidad_pqrs.utils import (
    mapping_data,
    define_service,
    define_f3,
    clean_text_TfIdf,
    optimize_threshold,
    create_probability_col
)


class TestMappingData:
    """Tests para mapping_data()"""
    
    def test_mapping_descontento_producto(self):
        """Descripciones pobres deben mapearse a DESCONTENTO CON EL PRODUCTO"""
        df = pd.DataFrame({
            'Proceso': ['WHATEVER'],
            'Causa': ['WHATEVER'],
            'Descripción': ['QUEJA'],  # Está en DESCONTENTO_PRODUCTO
            'Filtro 3': ['WHATEVER'],
            'Filtro 4': ['WHATEVER']
        })
        result = mapping_data(df)
        assert result.loc[0, 'Causa'] == 'DESCONTENTO CON EL PRODUCTO'
    

    def test_mapping_process_dict(self):
        """Procesos deben homologarse según PROCESS_DICT"""
        df = pd.DataFrame({
            'Proceso': ['GESTION DE PAGOS'],
            'Causa': ['WHATEVER'],
            'Descripción': ['WHATEVER'],
            'Filtro 3': ['WHATEVER'],
            'Filtro 4': ['WHATEVER']
        })
        result = mapping_data(df)
        assert result.loc[0, 'Proceso'] == 'RECAUDOS'
    

    def test_mapping_causes_dict(self):
        """Causas deben homologarse según CAUSES_DICT"""
        df = pd.DataFrame({
            'Proceso': ['WHATEVER'],
            'Causa': ['INADECUADA ATENCION DEL EMPLEADO SURA'],
            'Descripción': ['WHATEVER'],
            'Filtro 3': ['WHATEVER'],
            'Filtro 4': ['WHATEVER']
        })
        result = mapping_data(df)
        assert result.loc[0, 'Causa'] == 'INADECUADA ATENCION DEL PROVEEDOR/PRESTADOR'
    

    def test_filtro4_asistencia_salud(self):
        """Filtro 4 debe ser 'No identificado' si es NaN en ASISTENCIA SALUD"""
        df = pd.DataFrame({
            'Proceso': ['ASISTENCIA SALUD'],
            'Causa': ['WHATEVER'],
            'Descripción': ['WHATEVER'],
            'Filtro 3': ['WHATEVER'],
            'Filtro 4': [np.nan]
        })
        result = mapping_data(df)
        assert result.loc[0, 'Filtro 4'] == 'No identificado'
    

    def test_filtro4_other_processes(self):
        """Filtro 4 debe ser NaN si no es ASISTENCIA SALUD"""
        df = pd.DataFrame({
            'Proceso': ['WHATEVER'],
            'Causa': ['WHATEVER'],
            'Descripción': ['WHATEVER'],
            'Filtro 3': ['WHATEVER'],
            'Filtro 4': ['WHATEVER']
        })
        result = mapping_data(df)
        assert pd.isna(result.loc[0, 'Filtro 4'])



class TestDefineService:
    """Tests para define_service()"""
    
    def test_non_asistencia_returns_nan(self):
        """Procesos que no son ASISTENCIA SALUD deben retornar NaN"""
        row = pd.Series({
            'Proceso': 'WHATEVER',
            'Filtro 4': 'WHATEVER',
            'Causa': 'WHATEVER'
        })
        assert pd.isna(define_service(row))
    

    def test_filtro4_with_value(self):
        """Si hay Filtro 4, debe mapear desde dict.json"""

        row = pd.Series({
            'Proceso': 'ASISTENCIA SALUD',
            'Filtro 4': 'CONSULTA CON MEDICO GENERAL O ESPECIALISTA',
            'Causa': 'WHATEVER'
        })
        result = define_service(row)
        assert result in ['Consultas', 'TBD']
    

    def test_causa_with_incapacidad(self):
        """Si causa contiene INCAPACIDAD, debe retornar 'Orden'"""
        row = pd.Series({
            'Proceso': 'ASISTENCIA SALUD',
            'Filtro 4': np.nan,
            'Causa': 'DIFICULTAD TRANSCRIPCION DE INCAPACIDADES'
        })
        assert define_service(row) == 'Orden'
    
    
    def test_causa_with_medicamento(self):
        """Si causa contiene MEDICAMENTO, debe retornar 'Medicamentos'"""
        row = pd.Series({
            'Proceso': 'ASISTENCIA SALUD',
            'Filtro 4': np.nan,
            'Causa': 'PROBLEMAS CON ENTREGA DE MEDICAMENTOS'
        })
        assert define_service(row) == 'Medicamentos'
    

    def test_causa_with_red_medica(self):
        """Si causa contiene RED MEDICA, debe retornar 'Consultas'"""
        row = pd.Series({
            'Proceso': 'ASISTENCIA SALUD',
            'Filtro 4': np.nan,
            'Causa': 'INCONFORMIDAD CON LA RED MEDICA'
        })
        assert define_service(row) == 'Consultas'
    

    def test_no_filtro4_no_keywords(self):
        """Si no hay Filtro 4 ni keywords en causa, debe retornar 'TBD'"""
        row = pd.Series({
            'Proceso': 'ASISTENCIA SALUD',
            'Filtro 4': np.nan,
            'Causa': 'OTRA CAUSA GENERICA'
        })
        assert define_service(row) == 'TBD'



class TestDefineF3:
    """Tests para define_f3()"""
    
    def test_nan_input(self):
        """NaN debe retornar NaN"""
        assert pd.isna(define_f3(np.nan))
    
    def test_numeric_string(self):
        """Strings con números deben retornar 'Ciudad'"""
        assert define_f3('12345') == 'Ciudad'
        assert define_f3('Sede 123') == 'Ciudad'
    
    def test_text_only(self):
        """Strings sin números deben retornar 'Ente'"""
        assert define_f3('Hospital General') == 'Ente'
        assert define_f3('Clínica') == 'Ente'



class TestTextCleaning:
    """Tests para clean_text_TfIdf()"""
    
    def test_lowercase_conversion(self):
        """Debe convertir a minúsculas"""
        df = pd.DataFrame({
            'Descripción': ['QUEJA URGENTE SOBRE SERVICIO']
        })
        result = clean_text_TfIdf(df)
        assert result.loc[0, 'Descripción_TfIdf'].islower()
    

    def test_removes_punctuation(self):
        """Debe remover signos de puntuación"""
        df = pd.DataFrame({
            'Descripción': ['¡Hola! ¿Cómo estás? Bien, gracias.']
        })
        result = clean_text_TfIdf(df)
        text = result.loc[0, 'Descripción_TfIdf']
        assert '¡' not in text
        assert '?' not in text
        assert ',' not in text
    

    def test_creates_required_columns(self):
        """Debe crear las columnas esperadas"""
        df = pd.DataFrame({
            'Descripción': ['Texto de prueba']
        })
        result = clean_text_TfIdf(df)
        assert 'Descripción_TfIdf' in result.columns
    

    def test_removes_multiple_spaces(self):
        """Debe remover espacios múltiples"""
        df = pd.DataFrame({
            'Descripción': ['Texto    con     espacios    múltiples']
        })
        result = clean_text_TfIdf(df)
        assert '  ' not in result.loc[0, 'Descripción_TfIdf']


class TestOptimizeThreshold:
    
    def test_basic_optimization(self):
        """Test básico de optimización de umbrales"""
        df = pd.DataFrame({
            'clase_real': ['A', 'A', 'B', 'B', 'A', 'B'],
            'clase_predicha': ['A', 'A', 'A', 'B', 'B', 'B'],
            'prob_A': [0.9, 0.85, 0.6, 0.3, 0.4, 0.2],
            'prob_B': [0.1, 0.15, 0.4, 0.7, 0.6, 0.8]
        })
        
        result = optimize_threshold(df, threshold_error=0.12)
        
        assert len(result) == 2  # Dos clases
        assert all(col in result.columns for col in ['Clase', 'Umbral'])
        assert all(result['Umbral'] >= 0)
        assert all(result['Umbral'] <= 1)
    

    def test_threshold_respects_error_limit(self):
        """Los umbrales deben respetar el límite de error"""
        df = pd.DataFrame({
            'clase_real': ['A'] * 10 + ['B'] * 10,
            'clase_predicha': ['A'] * 15 + ['B'] * 5,
            'prob_A': [0.9] * 10 + [0.3] * 10,
            'prob_B': [0.1] * 10 + [0.7] * 10
        })
        
        threshold_error = 0.05
        result = optimize_threshold(df, threshold_error=threshold_error)
        
        # Los falsos positivos que pasan deben ser <= threshold_error
        fp_rates = result['Falsos que pasan (%)'].dropna()
        assert all(fp_rates <= threshold_error)
    
    def test_default_threshold_when_impossible(self):
        """Si no se puede cumplir el threshold_error, debe usar 0.8"""
        df = pd.DataFrame({
            'clase_real': ['A', 'A'],
            'clase_predicha': ['B', 'B'],  # Todo mal predicho
            'prob_A': [0.2, 0.3],
            'prob_B': [0.8, 0.7]
        })
        
        result = optimize_threshold(df, threshold_error=0.0)
        
        # Cuando es imposible cumplir, debería usar umbral por defecto
        assert all(result['Umbral'] == 0.8)


class TestCreateProbabilityCol:
    """Tests para create_probability_col()"""
    
    def test_creates_max_probability_column(self):
        """Debe crear columna con probabilidad máxima"""
        df = pd.DataFrame({
            'prob_A': [0.3, 0.7, 0.2],
            'prob_B': [0.5, 0.2, 0.6],
            'prob_C': [0.2, 0.1, 0.2]
        })
        
        result = create_probability_col(df, 'max_prob')
        
        assert 'max_prob' in result.columns
        assert result.loc[0, 'max_prob'] == 0.5
        assert result.loc[1, 'max_prob'] == 0.7
        assert result.loc[2, 'max_prob'] == 0.6
    

    def test_removes_prob_columns(self):
        """Debe remover columnas prob_*"""
        df = pd.DataFrame({
            'prob_A': [0.3],
            'prob_B': [0.7],
            'other_col': [1]
        })
        
        result = create_probability_col(df, 'max_prob')
        
        assert 'prob_A' not in result.columns
        assert 'prob_B' not in result.columns
        assert 'other_col' in result.columns
    

    def test_handles_single_probability(self):
        """Debe manejar caso con una sola probabilidad"""
        df = pd.DataFrame({
            'prob_A': [0.8]
        })
        
        result = create_probability_col(df, 'max_prob')
        
        assert result.loc[0, 'max_prob'] == 0.8