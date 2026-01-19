"""
Label Parser for Caltech Behavior Annotator annotation files
解析标注文件，支持 S1 (Behavior 3分类) 和 S2 (Aggression 7分类) 实验
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re


class AnnotationParser:
    """解析Caltech Behavior Annotator标注文件"""
    
    # S1 Behavior 标签定义 (3分类，排除base)
    BEHAVIOR_LABELS = {
        'aggression': 0,
        'social': 1,
        'nonsocial': 2,
    }
    
    # S2 Aggression 标签定义 (7分类，排除base)
    AGGRESSION_LABELS = {
        'lateralthreat': 0,
        'keepdown': 1,
        'clinch': 2,
        'uprightposture': 3,
        'freezing': 4,
        'bite': 5,
        'chase': 6,
    }
    
    # 类别名称映射
    AGGRESSION_NAMES = [
        'Lateral threat',
        'Keep down', 
        'Clinch',
        'Upright posture',
        'Freezing',
        'Bite',
        'Chase'
    ]
    
    BEHAVIOR_NAMES = ['Aggression', 'Social', 'Non-social']
    
    def __init__(self):
        pass
    
    def parse_annotation_file(self, file_path):
        """解析单个标注文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        s1_annotations = []
        s2_annotations = []
        
        # 查找S1部分
        s1_match = re.search(r'S1:\s+start\s+end\s+type\s*\n-+\n(.*?)(?=S2:|$)', content, re.DOTALL)
        if s1_match:
            s1_text = s1_match.group(1)
            for line in s1_text.strip().split('\n'):
                match = re.match(r'\s*(\d+)\s+(\d+)\s+(\w+)', line.strip())
                if match:
                    start, end, label = int(match.group(1)), int(match.group(2)), match.group(3).lower()
                    s1_annotations.append((start, end, label))
        
        # 查找S2部分
        s2_match = re.search(r'S2:\s+start\s+end\s+type\s*\n-+\n(.*?)$', content, re.DOTALL)
        if s2_match:
            s2_text = s2_match.group(1)
            for line in s2_text.strip().split('\n'):
                match = re.match(r'\s*(\d+)\s+(\d+)\s+(\w+)', line.strip())
                if match:
                    start, end, label = int(match.group(1)), int(match.group(2)), match.group(3).lower()
                    s2_annotations.append((start, end, label))
        
        return s1_annotations, s2_annotations
    
    def annotations_to_frame_labels(self, annotations, n_frames, label_map, exclude_labels=None):
        """将区间标注转换为逐帧标签"""
        if exclude_labels is None:
            exclude_labels = []
        
        frame_labels = np.full(n_frames, -1, dtype=np.int32)
        
        for start, end, label in annotations:
            label = label.lower()
            
            if label in exclude_labels:
                continue
            
            if label in label_map:
                start_idx = max(0, start - 1)
                end_idx = min(n_frames, end)
                frame_labels[start_idx:end_idx] = label_map[label]
        
        return frame_labels
    
    def create_behavior_labels(self, annot_file, n_frames):
        """
        创建S1 Behavior实验的标签 (3分类)
        排除base，保留 aggression(0), social(1), nonsocial(2)
        """
        s1_annotations, _ = self.parse_annotation_file(annot_file)
        
        label_map = {'aggression': 0, 'social': 1, 'nonsocial': 2}
        exclude_labels = ['base']  # 排除 base
        
        return self.annotations_to_frame_labels(s1_annotations, n_frames, label_map, exclude_labels)
    
    def create_aggression_labels(self, annot_file, n_frames):
        """
        创建S2 Aggression实验的标签 (7分类)
        排除base，保留7种攻击性行为细分类
        """
        _, s2_annotations = self.parse_annotation_file(annot_file)
        
        label_map = {
            'lateralthreat': 0,
            'keepdown': 1,
            'clinch': 2,
            'uprightposture': 3,
            'freezing': 4,
            'bite': 5,
            'chase': 6
        }
        exclude_labels = ['base']
        
        return self.annotations_to_frame_labels(s2_annotations, n_frames, label_map, exclude_labels)


def parse_all_annotations(annot_folder, n_frames_dict, experiment='behavior'):
    """解析所有标注文件"""
    parser = AnnotationParser()
    annot_folder = Path(annot_folder)
    all_labels = {}
    
    for annot_file in sorted(annot_folder.glob("*_annot.txt")):
        try:
            video_id = int(annot_file.stem.split('_')[0])
        except ValueError:
            print(f"  Warning: Cannot parse video ID from {annot_file.name}")
            continue
        
        if video_id not in n_frames_dict:
            print(f"  Warning: No frame count for video {video_id}")
            continue
        
        n_frames = n_frames_dict[video_id]
        
        if experiment == 'behavior':
            labels = parser.create_behavior_labels(annot_file, n_frames)
        else:
            labels = parser.create_aggression_labels(annot_file, n_frames)
        
        all_labels[video_id] = labels
        
        valid_count = np.sum(labels >= 0)
        print(f"  Video {video_id}: {valid_count}/{n_frames} valid frames ({100*valid_count/n_frames:.1f}%)")
    
    return all_labels


def get_class_names(experiment='behavior'):
    """获取类别名称列表"""
    if experiment == 'behavior':
        return AnnotationParser.BEHAVIOR_NAMES
    else:
        return AnnotationParser.AGGRESSION_NAMES


# ============== 测试代码 ==============
if __name__ == "__main__":
    test_content = """
Caltech Behavior Annotator - Annotation File

Configuration file:
base	z
aggression	a

S1:	start	end	type
-----------------------------
       1	943	base
       944	1142	nonsocial
       1143	1233	social
       1234	1288	aggression
  
S2:	start	end	type
-----------------------------
       1	1233	base
       1234	1288	lateralthreat
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='_annot.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    parser = AnnotationParser()
    s1, s2 = parser.parse_annotation_file(temp_file)
    
    print("S1 Annotations:")
    for ann in s1:
        print(f"  {ann}")
    
    print("\nS2 Annotations:")
    for ann in s2:
        print(f"  {ann}")
    
    # 测试标签转换
    print("\n--- Testing Behavior Labels (3-class) ---")
    labels = parser.create_behavior_labels(temp_file, 1500)
    for i, name in enumerate(AnnotationParser.BEHAVIOR_NAMES):
        count = np.sum(labels == i)
        print(f"  {i}: {name} = {count} frames")
    print(f"  Excluded (base/-1): {np.sum(labels == -1)} frames")
    
    import os
    os.unlink(temp_file)
    
    print("\n✓ Label parser test passed!")