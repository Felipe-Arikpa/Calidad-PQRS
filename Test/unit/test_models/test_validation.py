import pytest
import pandas as pd
import sys
from pathlib import Path
from Evaluation.validate_model import complaints_for_eval_causes, evaluate_log


class TestComplaintsForEvalCauses:
    """Tests para complaints_for_eval_causes()"""
    
    def test_filters_matching_processes(self):
        """Debe filtrar solo quejas donde proceso predicho = proceso real"""
        df = pd.DataFrame({
            'RAC_Process': ['A', 'A', 'B', 'A'],
            'Proceso_Sugerido': ['A', 'B', 'B', 'A'],
            'Causa': ['C1', 'C2', 'C3', 'C4']
        })
        
        result = complaints_for_eval_causes(df)
        assert len(result) == 3
        assert all(result['RAC_Process'] == result['Proceso_Sugerido'])
    

    def test_returns_empty_if_no_matches(self):
        """Debe retornar DataFrame vacío si no hay coincidencias"""
        df = pd.DataFrame({
            'RAC_Process': ['A', 'A', 'A'],
            'Proceso_Sugerido': ['B', 'B', 'B'],
            'Causa': ['C1', 'C2', 'C3']
        })
        
        result = complaints_for_eval_causes(df)
        assert len(result) == 0


class TestEvaluateLog:
    """Tests para evaluate_log()"""
    
    def test_warning_when_below_threshold(self):
        """Debe retornar 'warning' cuando scores están bajo umbral"""
        df = pd.DataFrame({
            'complaints_number': [100, 100, 100],
            'process_score': [0.3, 0.35, 0.32],  # Bajo umbral 0.4
            'causes_score': [0.6, 0.55, 0.58]
        })
        
        result = evaluate_log(df, umbral_process=0.4, umbral_causes=0.5)
        
        assert result == 'warning'
    
    def test_no_warning_when_above_threshold(self):
        """Debe retornar 'no warning' cuando scores están sobre umbral"""
        df = pd.DataFrame({
            'complaints_number': [100, 100, 100],
            'process_score': [0.8, 0.82, 0.85],
            'causes_score': [0.75, 0.78, 0.80]
        })
        
        result = evaluate_log(df, umbral_process=0.4, umbral_causes=0.5)
        
        assert result == 'no warning'
    
    def test_returns_none_with_insufficient_data(self):
        """Debe retornar None si hay menos de 3 registros con >90 quejas"""
        df = pd.DataFrame({
            'complaints_number': [100, 50],  # Solo 1 con >90
            'process_score': [0.8, 0.8],
            'causes_score': [0.8, 0.8]
        })
        
        result = evaluate_log(df, umbral_process=0.4, umbral_causes=0.5)
        
        assert result is None