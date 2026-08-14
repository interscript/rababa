"""Modern rababa — Arabic / Hebrew diacritization.

Modal-native training, ONNX inference, browser-deployable int8 models.

Public API:
    from rababa.datasets import load_tashkeela
    from rababa.models import build_student
    from rababa.training import train_supervised
    from rababa.export import export_student_onnx
    from rababa.evaluate import compute_der
"""

__version__ = "0.3.0.dev0"
