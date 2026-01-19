"""
Mouse Behavior Classification Package
"""

from .label_parser import AnnotationParser, parse_all_annotations, get_class_names
from .feature_extraction import DLCFeatureExtractor, get_feature_names
from .data_loader import (
    MouseBehaviorDataset,
    prepare_dataset,
    create_data_loaders,
    create_numpy_splits,
    create_video_level_splits,
    verify_split_integrity
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

__version__ = '1.0.1'
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
    'create_video_level_splits',
    'verify_split_integrity',
    # Models
    'BehaviorMLP',
    'BehaviorLSTM',
    'BehaviorCNN',
    'BehaviorTransformer',
    'get_pytorch_model',
    'get_sklearn_model',
    'compute_class_weights',
]
