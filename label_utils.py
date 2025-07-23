# label_utils.py

from constants import CLASS_MAP_BINARY, CLASS_MAP_4, CLASS_MAP_FULL

def get_num_classes(label_mode: str) -> int:
    if label_mode == "binary":
        return len(CLASS_MAP_BINARY)
    elif label_mode == "4class":
        return len(CLASS_MAP_4)
    elif label_mode == "full":
        return len(CLASS_MAP_FULL)
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

def get_num_labels(label_mode):
    # label modeに応じてクラス分けのラベルの種類数を数える
    if label_mode == '4class':
        labels = CLASS_MAP_4
    elif label_mode == 'binary':
        labels = CLASS_MAP_BINARY
    elif label_mode == 'full':
        labels = CLASS_MAP_FULL
    else:
        raise ValueError(f"Invalid label_mode: {label_mode}")    
    return len(labels)
