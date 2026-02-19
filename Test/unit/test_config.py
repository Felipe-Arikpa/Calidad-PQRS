import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from calidad_pqrs.config import (
    BASE_DIR, INPUT_DIR, MODEL_PROCESS_DIR, MODEL_CAUSES_DIR,
    OUTPUT_DIR, OUTPUT_ALERTS_DIR, OUTPUT_MONITORING_DIR,
    PROCESS_DICT, CAUSES_DICT, DESCONTENTO_PRODUCTO
)


class TestConfigPaths:
    """Tests para verificar que las rutas estén configuradas correctamente"""
    
    def test_base_dir_exists(self):
        """BASE_DIR debe ser un Path"""
        assert isinstance(BASE_DIR, Path)
    
    def test_input_dir_structure(self):
        """INPUT_DIR debe ser Path y apuntar a Input/"""
        assert isinstance(INPUT_DIR, Path)
        assert INPUT_DIR.name == "Input"
    
    def test_model_dirs_structure(self):
        """Directorios de modelos deben tener estructura correcta"""
        assert isinstance(MODEL_PROCESS_DIR, Path)
        assert isinstance(MODEL_CAUSES_DIR, Path)
        assert MODEL_PROCESS_DIR.name == "Process"
        assert MODEL_CAUSES_DIR.name == "Causes"
    
    def test_output_dirs_structure(self):
        """Directorios de output deben estar configurados"""
        assert isinstance(OUTPUT_DIR, Path)
        assert isinstance(OUTPUT_ALERTS_DIR, Path)
        assert isinstance(OUTPUT_MONITORING_DIR, Path)


class TestConfigDictionaries:
    """Tests para verificar diccionarios de configuración"""
    
    def test_process_dict_not_empty(self):
        """PROCESS_DICT no debe estar vacío"""
        assert len(PROCESS_DICT) > 0
        assert isinstance(PROCESS_DICT, dict)
    
    def test_causes_dict_not_empty(self):
        """CAUSES_DICT no debe estar vacío"""
        assert len(CAUSES_DICT) > 0
        assert isinstance(CAUSES_DICT, dict)
    
    def test_descontento_producto_is_set(self):
        """DESCONTENTO_PRODUCTO debe ser un set"""
        assert isinstance(DESCONTENTO_PRODUCTO, set)
        assert 'ADJUNTO QUEJA ESCRITA' in DESCONTENTO_PRODUCTO
        assert 'DERECHO DE PETICIÓN.' in DESCONTENTO_PRODUCTO
        assert 'VER ARCHIVO ADJUNTO' in DESCONTENTO_PRODUCTO



class TestConfigMappings:
    """Tests para verificar coherencia de los mapeos"""
    
    def test_process_dict_keys_are_strings(self):
        """Claves de PROCESS_DICT deben ser strings"""
        assert all(isinstance(k, str) for k in PROCESS_DICT.keys())
    
    def test_process_dict_values_are_strings(self):
        """Valores de PROCESS_DICT deben ser strings"""
        assert all(isinstance(v, str) for v in PROCESS_DICT.values())
    
    def test_causes_dict_keys_are_strings(self):
        """Claves de CAUSES_DICT deben ser strings"""
        assert all(isinstance(k, str) for k in CAUSES_DICT.keys())
    
    def test_causes_dict_values_are_strings(self):
        """Valores de CAUSES_DICT deben ser strings"""
        assert all(isinstance(v, str) for v in CAUSES_DICT.values())