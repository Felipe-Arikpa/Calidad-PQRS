import pytest
import pandas as pd
import sys
from pathlib import Path


class TestPreprocessingFunctions:
    """Tests para funciones de preprocesamiento"""
    
    def test_preprocessing_preserves_required_columns(self, sample_complaints_df):
        """El preprocesamiento debe preservar columnas requeridas"""
        from calidad_pqrs.utils import clean_text_TfIdf
        
        result = clean_text_TfIdf(sample_complaints_df)
        
        required_cols = ['Número del caso', 'Proceso', 'Causa', 
                        'Descripción', 'Fecha de apertura']
        for col in required_cols:
            assert col in result.columns
    
    def test_preprocessing_creates_tfidf_column(self, sample_complaints_df):
        """Debe crear columna Descripción_TfIdf"""
        from calidad_pqrs.utils import clean_text_TfIdf
        
        result = clean_text_TfIdf(sample_complaints_df)
        
        assert 'Descripción_TfIdf' in result.columns
        assert not result['Descripción_TfIdf'].isna().any()