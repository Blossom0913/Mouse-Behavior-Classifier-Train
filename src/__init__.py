"""
Mouse Behavior Classification Package
小鼠行为分类工具包
"""

from .label_parser import AnnotationParser, parse_all_annotations, get_class_names
from .feature_extraction import DLCFeatureExtractor, get_feature_names
from .data_loader import (
    MouseBehaviorDataset,
    prepare_dataset,
    create_data_loaders,
    create_numpy_splits
)
from .models import (
    BehaviorMLP,
    BehaviorLSTM,
    BehaviorCNN,
    BehaviorTransformer,
    get_pytorch_model,
    get_sklearn_model,
    compute_class_weights
)

__version__ = '1.0.0'
__author__ = 'Mouse Behavior Lab'

__all__ = [
    # Label Parser
    'AnnotationParser',
    'parse_all_annotations',
    'get_class_names',
    # Feature Extraction
    'DLCFeatureExtractor',
    'get_feature_names',
    # Data Loader
    'MouseBehaviorDataset',
    'prepare_dataset',
    'create_data_loaders',
    'create_numpy_splits',
    # Models
    'BehaviorMLP',
    'BehaviorLSTM',
    'BehaviorCNN',
    'BehaviorTransformer',
    'get_pytorch_model',
    'get_sklearn_model',
    'compute_class_weights',
]
