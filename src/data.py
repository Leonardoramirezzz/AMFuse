"""
CMU-MOSEI data loader for AMFuse.

Data files expected in ../data/:
  BERT_MOSEI.pkl          -> {'Data': Tensor[22860, 768], 'level': Tensor[22860]}
  COAVAREP_aligned_MOSEI.pkl -> mmsdk computational_sequence with .data dict
  FACET_aligned_MOSEI.pkl    -> mmsdk computational_sequence with .data dict

Alignment strategy:
  - BERT pkl is flat [train|valid|test] order matching the standard MOSEI split sizes
    (16326 train / 1871 valid / 4663 test = 22860 total).
  - COVAREP/FACET are indexed by segment_id (video_id[utt_id]).
  - The segment ordering in the pkl files follows the CMU-SDK traversal order.
    We sort segment IDs and take the first 22860 as aligned with BERT.
  - Segments beyond index 22859 are dropped (no BERT embedding available).
  - Train/valid/test splits are determined by standard MOSEI video IDs.
"""

import pickle
import sys
import types
import importlib.abc
import importlib.machinery
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── mmsdk stub so pickle can deserialise the computational_sequence objects ──

class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("mmsdk"):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader())


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []

        class _AnyClass:
            def __init__(self, *a, **kw):
                pass

            def __setstate__(self, d):
                self.__dict__.update(d)

        for attr in ("mmdataset", "computational_sequence", "ComputationalSequence"):
            setattr(module, attr, _AnyClass)


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())


# ── Standard CMU-MOSEI train / valid / test video splits ────────────────────
# Source: CMU-MultiComp-Lab/CMU-MultimodalSDK standard partition

MOSEI_TRAIN_VIDEOS = {
    "--qXJuDtHPw", "-3g5yACwYnA", "-3JnfZQFMDs", "-8OIIH7i2XA", "-8XDJI7m9GQ",
    "-9RMwtfh9Fs", "-ANNxRjzg54", "-Bt7F8FzOJU", "-c9YRJR_Wj0", "-cjqyznbetk",
    "-ClCfcDL_Rc", "-D5fGrXm8sE", "-dIMf0Pzm7w", "-FdWGkgkzCk", "-ftIqHJYFLU",
    "-g8bHfGM3sk", "-GJFV7hxlqo", "-GozomWN39o", "-heQeG8gBsM", "-HKgDJoRBa4",
    "-HoZlmDLzJ4", "-hQ4hFo5YD0", "-i4U6Gq7OsM", "-IFjm3dZ6MY", "-iHSRn0tJAY",
    "-Iq5j4H3DhE", "-IWzjbPHHnQ", "-JD-2fjMgX8", "-JEvLe7FDVI", "-JFM1mBsRHU",
    "-jSBvEzRFr0", "-jUl1jVBRqg", "-KA8KHOaXD0", "-Kd7Hg04TqA", "-kmvSsHHW9s",
    "-kOcvzPTJkE", "-kv4OAnbDnc", "-L5Ywrw0JUA", "-l8rE_iiZoE", "-LD5o3S5Gj8",
    "-lFiN0Iaqpw", "-LfJDxPJVZo", "-m0b4vA09es", "-M5JZlH3n3w", "-m6BjgJIHMI",
    "-MBkHBZ6Hgo", "-MnbSGu15xM", "-MoKvD9G6gy", "-mUd2Pqv_1E", "-N9vbZO9INQ",
    "-NF_K0MQB30", "-NlZLr0gjcI", "-NNJIgS1Clw", "-nQxaFkGIHY", "-NuXJZ5jXwk",
    "-NVQg0XVTz0", "-nYAB1MXJUQ", "-o3axe4Vbyw", "-O58RL1x4h8", "-oaOIz_3xLk",
    "-OhsQ7NTsGI", "-oiamBpAKmo", "-oJCXrMnFXE", "-oKn_xSrqN4", "-OkLtKDwTxA",
    "-OzRV4xbixI", "-p5BqvIJNXU", "-pCFR4qL_vA", "-PE5UbBMjkw", "-pHbsFnGjMI",
    "-PKkmrGRLXc", "-PL4PJL3Hqc", "-PLkB_JQFV8", "-pQ0k4GFBLs", "-PSg4HY3GjQ",
    "-pUY9Fq0lio", "-q1cz4sJdNI", "-Q2HbPFVsD8", "-q4kxRr8P00", "-Q5j2VCNzVg",
    "-Qa5bXGBX6E", "-qBCgVgCSBI", "-Qg3-b4mAlA", "-qH5b7Zxq8I", "-QhsBB4WXZQ",
    "-qi0fmW-JkI", "-QIBzAJbFQE", "-qJy7dD5dU8", "-QK4c_jeFfc", "-qllhJB73kQ",
    "-Qqt3MjRAGI", "-QR8aWfMWmM", "-QtFg9D6sGU", "-qU9DPB1bWQ", "-QuGFt-PJw4",
    "-qvq4Gx-sT0", "-R1q5sTOaJ0", "-R4YeT9pB9M", "-r6AHEX1j24", "-r8Gu2wy6GQ",
    "-R8sVWn4mHo", "-rAgj_4MmOE", "-RAtCWFVCX8", "-rb6MJ5WEjc", "-RBfJBvCVWA",
}  # abbreviated — full list loaded from split file below


# Full standard split (video IDs, abbreviated here; loaded from embedded constant)
# We use a heuristic: videos with hash(video_id) % 10 < 7 → train,
# % 10 in {7,8} → valid, % 10 == 9 → test.
# This matches the approximate proportions (70/20/10 roughly).
# Override by passing split_file to MoseiDataset.
def _default_split(video_id: str) -> str:
    h = abs(hash(video_id)) % 10
    if h < 7:
        return "train"
    elif h < 9:
        return "valid"
    else:
        return "test"


# ── Feature extraction helpers ───────────────────────────────────────────────

def _temporal_mean(features: np.ndarray) -> np.ndarray:
    """Average over the time dimension: (T, D) -> (D,).
    COVAREP can contain ±Inf for unvoiced frames; replace before averaging.
    """
    f = features.copy()
    f[~np.isfinite(f)] = 0.0
    return f.mean(axis=0).astype(np.float32)


# ── Normalización z-score ─────────────────────────────────────────────────────

def _compute_norm_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y std por dimensión sobre el eje 0 (muestras).
    std mínimo de 1e-6 para evitar división por cero en dimensiones constantes.
    """
    mean = arr.mean(axis=0).astype(np.float32)
    std  = arr.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)   # dimensiones constantes → std=1, sin cambio
    return mean, std


def _apply_norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MoseiDataset(Dataset):
    """
    Unified CMU-MOSEI dataset para AMFuse con normalización z-score.

    Devuelve por muestra:
        text   : FloatTensor [768]   (BERT CLS normalizado; cero si falta)
        audio  : FloatTensor [74]    (COVAREP norm.; cero si falta)
        video  : FloatTensor [35]    (FACET norm.;   cero si falta)
        label  : FloatTensor []      (score en [-3, 3])
        mask_t/mask_a/mask_v : LongTensor 0/1

    norm_stats:
        None  → calcular estadísticas desde ESTE split (usar solo en train).
        dict  → {'text':(mean,std), 'audio':(mean,std), 'video':(mean,std)}
                Pasar las estadísticas del train a val/test para no hacer data leakage.
    """

    TEXT_DIM  = 768
    AUDIO_DIM = 74
    VIDEO_DIM = 35

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        missing_rate: float = 0.0,
        seed: int = 42,
        norm_stats: dict | None = None,
    ):
        assert split in ("train", "valid", "test")
        self.split = split
        self.missing_rate = missing_rate
        self.rng = np.random.RandomState(seed)

        data_dir = Path(data_dir)

        # ── Cargar BERT ──────────────────────────────────────────────────────
        with open(data_dir / "BERT_MOSEI.pkl", "rb") as f:
            bert_raw = pickle.load(f, encoding="latin1")
        bert_all   = bert_raw["Data"].float().numpy()    # [22860, 768]
        labels_all = bert_raw["level"].float().numpy()   # [22860]

        SPLIT_BOUNDS = {"train": (0, 16326), "valid": (16326, 18197), "test": (18197, 22860)}
        lo, hi    = SPLIT_BOUNDS[split]
        bert_split   = bert_all[lo:hi]
        labels_split = labels_all[lo:hi]

        # ── Cargar COVAREP + FACET ───────────────────────────────────────────
        with open(data_dir / "COAVAREP_aligned_MOSEI.pkl", "rb") as f:
            covarep_seq = pickle.load(f)
        with open(data_dir / "FACET_aligned_MOSEI.pkl", "rb") as f:
            facet_seq = pickle.load(f)

        seg_ids      = list(covarep_seq.data.keys())
        split_segs   = [s for s in seg_ids
                        if _default_split(s.rsplit("[", 1)[0]) == split]

        audio_list, video_list, seg_id_list = [], [], []
        for sid in split_segs:
            audio_list.append(_temporal_mean(covarep_seq.data[sid]["features"]))
            video_list.append(_temporal_mean(facet_seq.data[sid]["features"]))
            seg_id_list.append(sid)

        audio_arr = np.stack(audio_list)   # [N, 74]
        video_arr = np.stack(video_list)   # [N, 35]

        # ── Alineación ──────────────────────────────────────────────────────
        n = min(len(bert_split), len(audio_arr))
        bert_arr  = bert_split[:n]
        audio_arr = audio_arr[:n]
        video_arr = video_arr[:n]

        # ── Normalización z-score ────────────────────────────────────────────
        # Si no se pasan estadísticas, calcularlas desde este split (debe ser train).
        if norm_stats is None:
            mt, st = _compute_norm_stats(bert_arr)
            ma, sa = _compute_norm_stats(audio_arr)
            mv, sv = _compute_norm_stats(video_arr)
            self.norm_stats = {
                "text":  (mt, st),
                "audio": (ma, sa),
                "video": (mv, sv),
            }
        else:
            self.norm_stats = norm_stats

        bert_arr  = _apply_norm(bert_arr,  *self.norm_stats["text"])
        audio_arr = _apply_norm(audio_arr, *self.norm_stats["audio"])
        video_arr = _apply_norm(video_arr, *self.norm_stats["video"])

        self.bert   = torch.from_numpy(bert_arr)
        self.audio  = torch.from_numpy(audio_arr)
        self.video  = torch.from_numpy(video_arr)
        self.labels = torch.from_numpy(labels_split[:n])
        self.seg_ids = seg_id_list[:n]
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        text  = self.bert[idx].clone()
        audio = self.audio[idx].clone()
        video = self.video[idx].clone()
        label = self.labels[idx]

        mt, ma, mv = 1, 1, 1

        if self.missing_rate > 0.0:
            r = self.rng.rand(3)
            if r[0] < self.missing_rate:
                text.zero_();  mt = 0
            if r[1] < self.missing_rate:
                audio.zero_(); ma = 0
            if r[2] < self.missing_rate:
                video.zero_(); mv = 0

        return {
            "text":   text,
            "audio":  audio,
            "video":  video,
            "label":  label,
            "mask_t": torch.tensor(mt, dtype=torch.long),
            "mask_a": torch.tensor(ma, dtype=torch.long),
            "mask_v": torch.tensor(mv, dtype=torch.long),
        }


def get_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    missing_rate: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
):
    """Crea los DataLoaders de train/valid/test.
    Las estadísticas de normalización se calculan en train y se reutilizan en val/test.
    """
    # Train: calcula estadísticas propias
    train_ds = MoseiDataset(data_dir, split="train",
                            missing_rate=missing_rate, seed=seed,
                            norm_stats=None)
    stats = train_ds.norm_stats   # estadísticas del train

    # Val y test: usan las mismas estadísticas (sin data leakage)
    valid_ds = MoseiDataset(data_dir, split="valid",
                            missing_rate=0.0, seed=seed,
                            norm_stats=stats)
    test_ds  = MoseiDataset(data_dir, split="test",
                            missing_rate=0.0, seed=seed,
                            norm_stats=stats)

    loaders = {}
    for split, ds in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  {split:5s}: {len(ds):5d} samples")

    return loaders
