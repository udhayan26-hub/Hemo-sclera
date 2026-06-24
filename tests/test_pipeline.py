import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add root directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import predict_sample

class TestPipeline(unittest.TestCase):

    @patch('pipeline.isolate_sclera')
    @patch('pipeline.cv2.imread')
    @patch('pipeline.analyze_jaundice_sclera')
    @patch('joblib.load')
    def test_predict_sample_jaundice_success(self, mock_joblib_load, mock_analyze, mock_imread, mock_isolate):
        # Setup mocks
        mock_isolate.return_value = (None, MagicMock())
        mock_imread.return_value = MagicMock()
        mock_analyze.return_value = 15.5
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.15, 0.85]]
        mock_joblib_load.return_value = mock_model
        
        # We patch sys.stdout to capture prints
        with patch('sys.stdout') as mock_stdout:
            predict_sample('dummy_path.png', 'jaundice')
            # Verify mock model loading and prediction was called
            mock_joblib_load.assert_called_once_with('jaundice_lr_model.pkl')

    @patch('pipeline.isolate_sclera')
    def test_predict_sample_anemia_message(self, mock_isolate):
        with patch('sys.stdout') as mock_stdout:
            predict_sample('dummy_path.png', 'anemia')
            # Verify it does not call isolate_sclera
            mock_isolate.assert_not_called()

if __name__ == '__main__':
    unittest.main()
