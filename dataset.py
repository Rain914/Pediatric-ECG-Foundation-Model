import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb
from scipy import signal


# ============================================================
# Label utilities
# ============================================================
def parse_label_cell(x):
    """Parse one Label cell into a clean Python list of label strings.

    Supported examples:
        ['NORM', 'SARRH']
        "['NORM', 'SARRH']"
        'NORM;SARRH'
        'NORM|SARRH'
        'NORM'
    """

    def clean_label(v):
        return str(v).strip().strip("'").strip('"').strip()

    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [clean_label(v) for v in x if clean_label(v)]

    if pd.isna(x):
        return []

    x = str(x).strip()
    if not x:
        return []

    try:
        value = ast.literal_eval(x)
        if isinstance(value, (list, tuple, set)):
            return [clean_label(v) for v in value if clean_label(v)]
        value = clean_label(value)
    except (ValueError, SyntaxError):
        value = clean_label(x)

    if ';' in value:
        return [clean_label(v) for v in value.split(';') if clean_label(v)]
    if '|' in value:
        return [clean_label(v) for v in value.split('|') if clean_label(v)]

    return [value] if value else []


def process_labels(
    train_label_path,
    test_label_path=None,
    saved_dir='.',
    fixed_label_set=None,
    label_column='Label',
):
    """Create multi-hot label matrices from CSV files.

    Parameters
    ----------
    train_label_path : str
        Training CSV path.
    test_label_path : str or None
        Optional test CSV path.
    saved_dir : str
        Directory used to save generated .npy matrices and label mapping.
    fixed_label_set : sequence[str] or None
        If provided, this exact label order is used. This is recommended for
        sample-efficiency experiments so 1/10/25/50/100% runs use the same
        output dimension and class order.
    label_column : str
        Name of the label column.

    Returns
    -------
    n_classes, label_set, train_npy_path, test_npy_path
    """
    os.makedirs(saved_dir, exist_ok=True)

    train_df = pd.read_csv(train_label_path).reset_index(drop=True)

    if label_column not in train_df.columns:
        raise ValueError(f"训练 CSV 中不存在列 {label_column!r}")

    train_lists = train_df[label_column].apply(parse_label_cell)

    empty_train = train_lists.map(len).eq(0)
    if empty_train.any():
        bad_rows = np.flatnonzero(empty_train.to_numpy())[:10].tolist()
        raise ValueError(
            f"训练集存在 {int(empty_train.sum())} 行空/无法解析的 Label，"
            f"示例行索引: {bad_rows}。请先修正数据，不要静默删除样本。"
        )

    if fixed_label_set is None:
        label_set = sorted({lab for labs in train_lists for lab in labs if lab})
    else:
        label_set = [str(x) for x in fixed_label_set]
        if len(label_set) != len(set(label_set)):
            raise ValueError('fixed_label_set 中存在重复标签')

    if not label_set:
        raise ValueError('未检测到任何有效标签')

    label_to_idx = {lab: i for i, lab in enumerate(label_set)}
    n_classes = len(label_set)

    def build_matrix(df, label_lists, prefix):
        matrix = np.zeros((len(df), n_classes), dtype=np.float32)
        unknown = set()

        for i, labs in enumerate(label_lists):
            for lab in labs:
                if lab in label_to_idx:
                    matrix[i, label_to_idx[lab]] = 1.0
                else:
                    unknown.add(lab)

        if unknown:
            print(
                f"[process_labels] {prefix} 中发现 {len(unknown)} 个不在标签词表中的标签，"
                f"这些标签不会写入矩阵。示例: {sorted(unknown)[:10]}"
            )

        path = os.path.join(saved_dir, f'{prefix}_multi_labels.npy')
        np.save(path, matrix)
        return path

    train_npy = build_matrix(train_df, train_lists, 'train_sampled')

    test_npy = None
    if test_label_path:
        test_df = pd.read_csv(test_label_path).reset_index(drop=True)
        if label_column not in test_df.columns:
            raise ValueError(f"测试 CSV 中不存在列 {label_column!r}")
        test_lists = test_df[label_column].apply(parse_label_cell)

        empty_test = test_lists.map(len).eq(0)
        if empty_test.any():
            bad_rows = np.flatnonzero(empty_test.to_numpy())[:10].tolist()
            raise ValueError(
                f"测试集存在 {int(empty_test.sum())} 行空/无法解析的 Label，"
                f"示例行索引: {bad_rows}。"
            )

        test_npy = build_matrix(test_df, test_lists, 'test_sampled')

    # Save the exact class order used by this run.
    mapping_df = pd.DataFrame({
        'class_index': np.arange(n_classes, dtype=int),
        'label_name': label_set,
    })
    mapping_df.to_csv(
        os.path.join(saved_dir, 'label_mapping.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print(f'检测到标签种类数: {n_classes}')
    print(f'标签映射示例: {list(label_to_idx.items())[:10]}')

    return n_classes, label_set, train_npy, test_npy


# ============================================================
# Shared ECG preprocessing
# ============================================================
def _ensure_channel_first_12lead(ecg):
    """Convert one ECG to [12, Length] without silently guessing invalid data."""
    ecg = np.asarray(ecg)

    if ecg.ndim != 2:
        raise ValueError(f'单条 ECG 必须为二维数组，实际 shape={ecg.shape}')

    if ecg.shape[0] == 12 and ecg.shape[1] != 12:
        return ecg

    if ecg.shape[1] == 12 and ecg.shape[0] != 12:
        return ecg.T

    if ecg.shape[0] == 12 and ecg.shape[1] == 12:
        raise ValueError('ECG shape=[12, 12]，无法判断哪一维为导联维')

    raise ValueError(f'无法识别 12 导联维度，实际 shape={ecg.shape}')


def preprocess_12lead_ecg(ecg, fs=500.0):
    """Shared preprocessing for both WFDB and NPY data.

    Pipeline:
        [12, Length]
        -> NaN/Inf handling
        -> 50 Hz notch filter
        -> 0.5-75 Hz band-pass filter
        -> per-lead Z-score
        -> float32
    """
    ecg = _ensure_channel_first_12lead(ecg).astype(np.float32, copy=False)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    if ecg.shape[-1] < 32:
        raise ValueError(f'ECG 长度过短，无法稳定滤波: {ecg.shape[-1]}')

    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    ecg = signal.filtfilt(b_notch, a_notch, ecg, axis=-1)

    b_band, a_band = signal.butter(
        4,
        [0.5, 75.0],
        btype='bandpass',
        fs=fs,
    )
    ecg = signal.filtfilt(b_band, a_band, ecg, axis=-1)

    mean = np.mean(ecg, axis=-1, keepdims=True)
    std = np.std(ecg, axis=-1, keepdims=True)
    ecg = (ecg - mean) / (std + 1e-8)

    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    return ecg.astype(np.float32, copy=False)


# ============================================================
# Unified classification dataset
# ============================================================
class PediatricECGDataset(Dataset):
    """Unified 12-lead pediatric ECG classification Dataset.

    source_type='wfdb':
        data_path = WFDB root directory
        csv_path  = CSV containing Imagefilename

    source_type='npy':
        data_path = .npy ECG file, shape [N, Length, 12] or [N, 12, Length]
        csv_path  = not required

    Both modes use exactly the same preprocess_12lead_ecg() pipeline.
    """

    def __init__(
        self,
        data_path,
        label_path,
        source_type,
        csv_path=None,
        fs=500.0,
        image_column='Imagefilename',
    ):
        self.source_type = str(source_type).lower()
        self.data_path = data_path
        self.fs = float(fs)
        self.image_column = image_column
        self.labels = np.load(label_path)

        if self.labels.ndim not in (1, 2):
            raise ValueError(f'不支持的标签 shape: {self.labels.shape}')

        if self.source_type == 'npy':
            self.ecg_data = np.load(data_path, mmap_mode='r')
            if self.ecg_data.ndim != 3:
                raise ValueError(
                    f'NPY ECG 必须为三维 [N,L,12] 或 [N,12,L]，实际 {self.ecg_data.shape}'
                )

            if self.ecg_data.shape[1] != 12 and self.ecg_data.shape[2] != 12:
                raise ValueError(
                    f'NPY 中未找到12导联维度，实际 shape={self.ecg_data.shape}'
                )

            n_samples = len(self.ecg_data)
            self.records = None

        elif self.source_type == 'wfdb':
            if not csv_path:
                raise ValueError("source_type='wfdb' 时必须提供 csv_path")

            self.records = pd.read_csv(csv_path).reset_index(drop=True)

            if image_column not in self.records.columns:
                raise ValueError(
                    f'CSV 中不存在 {image_column!r}，当前列: {list(self.records.columns)}'
                )

            # Do not drop rows here. Label matrix and CSV must remain aligned.
            if self.records[image_column].isna().any():
                raise ValueError(
                    f'{image_column} 存在 {int(self.records[image_column].isna().sum())} 个空值'
                )

            self.ecg_data = None
            n_samples = len(self.records)

        else:
            raise ValueError("source_type 只能是 'wfdb' 或 'npy'")

        if n_samples != len(self.labels):
            raise ValueError(
                f'ECG 样本数 ({n_samples}) 与标签样本数 ({len(self.labels)}) 不一致。'
                '请确保 label_path 由同一份、同一行顺序的 CSV/数据生成。'
            )

        print(
            f'PediatricECGDataset loaded: source={self.source_type}, '
            f'N={n_samples}, labels={self.labels.shape}'
        )

    def __len__(self):
        return len(self.labels)

    def _load_ecg(self, idx):
        if self.source_type == 'npy':
            return np.asarray(self.ecg_data[idx])

        filename = str(self.records.iloc[idx][self.image_column])
        record_name = filename[:-4] if filename.upper().endswith('.ECG') else filename
        record_path = os.path.join(self.data_path, record_name)

        ecg, meta = wfdb.rdsamp(record_path)

        wfdb_fs = meta.get('fs', None)
        if wfdb_fs is not None and not np.isclose(float(wfdb_fs), self.fs):
            raise ValueError(
                f'{filename} 采样率为 {wfdb_fs} Hz，但 Dataset 配置 fs={self.fs} Hz。'
                '请先统一采样率或传入正确 fs。'
            )

        if ecg.ndim != 2 or ecg.shape[1] != 12:
            raise ValueError(
                f'{filename} 不是期望的12导联数据，shape={ecg.shape}, '
                f"sig_name={meta.get('sig_name', 'Unknown')}"
            )

        return ecg

    def __getitem__(self, idx):
        ecg = self._load_ecg(idx)
        ecg = preprocess_12lead_ecg(ecg, fs=self.fs)

        label = self.labels[idx]
        if self.labels.ndim == 1:
            label_tensor = torch.tensor([label], dtype=torch.float32)
        else:
            label_tensor = torch.as_tensor(label, dtype=torch.float32)

        ecg_tensor = torch.from_numpy(ecg)
        return ecg_tensor, label_tensor

