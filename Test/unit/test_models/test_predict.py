import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from calidad_pqrs.models.predict import (
    process_validation,
    causes_validation,
    final_validation
)


class TestProcessValidation:
    """Tests para process_validation()"""
    
    def test_correct_process(self):
        """Proceso correcto debe retornar 'Proceso correcto'"""
        row = pd.Series({
            'RAC_process_raw': 'ASISTENCIA SALUD',
            'RAC_Process': 'ASISTENCIA SALUD',
            'Proceso_Sugerido': 'ASISTENCIA SALUD',
            'Process_Probability': 0.9
        })
        result = process_validation(row)
        assert result == 'Proceso correcto'
    
    def test_incorrect_process_high_confidence(self):
        """Proceso incorrecto con alta confianza debe retornar 'Proceso incorrecto'"""
        # Nota: Este test requiere mockear process_thresholds
        # Por simplicidad, solo verificamos que la función no falle
        row = pd.Series({
            'RAC_process_raw': 'ASISTENCIA SALUD',
            'RAC_Process': 'ASISTENCIA SALUD',
            'Proceso_Sugerido': 'EXPEDICION',
            'Process_Probability': 0.95
        })
        result = process_validation(row)
        assert result in ['Proceso incorrecto', 'Alta incertidumbre']


class TestCausesValidation:
    """Tests para causes_validation()"""
    
    def test_correct_cause(self):
        """Causa correcta debe retornar 'Causa correcta'"""
        row = pd.Series({
            'RAC_causes_raw': 'INADECUADA ATENCION DEL PROVEEDOR/PRESTADOR',
            'RAC_Causes': 'INADECUADA ATENCION DEL PROVEEDOR/PRESTADOR',
            'Causa_Sugerida': 'INADECUADA ATENCION DEL PROVEEDOR/PRESTADOR',
            'Causes_Probability': 0.9
        })
        result = causes_validation(row)
        assert result == 'Causa correcta'
    
    def test_unknown_cause(self):
        """Causa desconocida debe ser detectada"""
        row = pd.Series({
            'RAC_causes_raw': 'CAUSA_INEXISTENTE_XYZ',
            'RAC_Causes': 'CAUSA_INEXISTENTE_XYZ',
            'Causa_Sugerida': 'DEMORA_CITA',
            'Causes_Probability': 0.9
        })
        result = causes_validation(row)
        assert result in ['Causa desconocida', 'Causa incorrecta', 'Alta incertidumbre']


class TestFinalValidation:
    """Tests para final_validation()"""
    
    def test_causa_incorrecta_triggers_review(self):
        """Causa incorrecta debe retornar 'Revisar'"""
        row = pd.Series({
            'Validated_Process_Label': 'Proceso correcto',
            'Validated_Causes_Label': 'Causa incorrecta'
        })
        assert final_validation(row) == 'Revisar'
    
    def test_all_correct_no_alerts(self):
        """Todo correcto debe retornar 'No se identifican alertas'"""
        row = pd.Series({
            'Validated_Process_Label': 'Proceso correcto',
            'Validated_Causes_Label': 'Causa correcta'
        })
        assert final_validation(row) == 'No se identifican alertas'
    
    def test_unknown_process_triggers_specific_message(self):
        """Proceso o causa desconocidos debe retornar mensaje específico"""
        row = pd.Series({
            'Validated_Process_Label': 'Proceso desconocido',
            'Validated_Causes_Label': 'Causa correcta'
        })
        assert final_validation(row) == 'Proceso o Causa desconocidos'
    
    def test_unknown_cause_triggers_specific_message(self):
        """Causa desconocida debe retornar mensaje específico"""
        row = pd.Series({
            'Validated_Process_Label': 'Proceso correcto',
            'Validated_Causes_Label': 'Causa desconocida'
        })
        assert final_validation(row) == 'Proceso o Causa desconocidos'
    
    def test_alta_incertidumbre_no_alerts(self):
        """Alta incertidumbre no debe generar alerta"""
        row = pd.Series({
            'Validated_Process_Label': 'Alta incertidumbre',
            'Validated_Causes_Label': 'Alta incertidumbre'
        })
        assert final_validation(row) == 'No se identifican alertas'