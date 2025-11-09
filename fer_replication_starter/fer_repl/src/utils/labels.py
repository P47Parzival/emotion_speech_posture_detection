import os, re
from pathlib import Path

# canonical order used for reports
EMOTION_ORDER = ["Anger","Contempt","Disgust","Fear","Happy","Neutral","Sadness","Surprise"]

# mapping variations found in filenames/folders
EMOTION_MAP = {
    "anger":"Anger","angry":"Anger","1":"Anger",
    "contempt":"Contempt","contemptuous":"Contempt","2":"Contempt",
    "disgust":"Disgust","disgusted":"Disgust","3":"Disgust",
    "fear":"Fear","fearful":"Fear","4":"Fear",
    "happy":"Happy","happiness":"Happy","5":"Happy",
    "neutral":"Neutral","0":"Neutral",
    "sad":"Sadness","sadness":"Sadness","6":"Sadness",
    "surprise":"Surprise","surprised":"Surprise","7":"Surprise"
}

def _from_folder(path: str):
    parts = Path(path).parts
    for p in parts[::-1]:
        k = p.lower()
        if k in EMOTION_MAP:
            return EMOTION_MAP[k]
    return None

def _from_ckplus_style(filename: str):
    # expects *_N.* where N in 1..7
    m = re.search(r'_(\d)(?:\.[^.]+)?$', filename)
    if m:
        digit = m.group(1)
        if digit in EMOTION_MAP:
            return EMOTION_MAP[digit]
    return None

def _from_keywords(filename: str):
    k = Path(filename).stem.lower()
    for key in EMOTION_MAP.keys():
        if key in k:
            return EMOTION_MAP[key]
    return None

def infer_label_from_path(path: str):
    # Try folder-based, then CK+ suffix, then keywords
    lbl = _from_folder(path)
    if lbl: return lbl
    fn = os.path.basename(path)
    lbl = _from_ckplus_style(fn)
    if lbl: return lbl
    return _from_keywords(fn)

def label_to_idx(lbl: str) -> int:
    return EMOTION_ORDER.index(lbl)

def idx_to_label(idx: int) -> str:
    return EMOTION_ORDER[idx]
