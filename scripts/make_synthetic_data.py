"""Create FAKE data with the same columns as GTZAN features_30_sec.csv.

ONLY for testing the pipeline before the real dataset is downloaded.
Never use these numbers in your report.
Run from the project root:  python -m scripts.make_synthetic_data
"""
import numpy as np
import pandas as pd
from src.config import SYNTHETIC_CSV_PATH, RANDOM_STATE

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
BASE = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff",
        "zero_crossing_rate", "harmony", "perceptr"]

rng = np.random.default_rng(RANDOM_STATE)
rows = []
for gi, g in enumerate(GENRES):
    shift = rng.normal(0, 1, 60)           # genre signature
    for k in range(100):
        r = {"filename": f"{g}.{k:05d}.wav", "length": 661794}
        i = 0
        for b in BASE:
            r[f"{b}_mean"] = 0.3 + 0.05 * gi + 0.1 * shift[i] + rng.normal(0, 0.12)
            r[f"{b}_var"] = abs(0.02 + 0.01 * shift[i + 1] + rng.normal(0, 0.01))
            i += 2
        r["tempo"] = 100 + 5 * gi + rng.normal(0, 18)
        for m in range(1, 21):
            r[f"mfcc{m}_mean"] = shift[m] * 8 + rng.normal(0, 14)
            r[f"mfcc{m}_var"] = abs(60 + shift[m + 20] * 15 + rng.normal(0, 25))
        r["label"] = g
        rows.append(r)

df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df.to_csv(SYNTHETIC_CSV_PATH, index=False)
print("Wrote", SYNTHETIC_CSV_PATH, df.shape)
