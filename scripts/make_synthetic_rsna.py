"""Writes a tiny synthetic RSNA-knee-like dataset (CSV + minimal DICOM) for CPU smoke tests of the notebooks.

    python scripts/make_synthetic_rsna.py OUT_DIR

The images are random blobs; they carry no signal. This only exercises shapes, masks, I/O and the submission format.
"""
import sys, pathlib, uuid
import numpy as np, pandas as pd, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PHRASE = {"ACL": "ACL tear", "MCL": "MCL sprain", "Medial Meniscus": "medial meniscus tear",
          "Lateral Meniscus": "lateral meniscus tear", "Effusion": "joint effusion", "Baker's": "Baker's cyst",
          "Contusion": "bone contusion", "Fracture": "fracture"}
AX = {"Sagittal": ([0, 1, 0, 0, 0, -1], 2), "Coronal": ([1, 0, 0, 0, 0, -1], 1), "Axial": ([1, 0, 0, 0, 1, 0], 0)}

def write_series(root, study, plane, fluid, rng, n, size=96, spacing=0.4):
    sid = generate_uid(); d = root / study / sid; d.mkdir(parents=True, exist_ok=True)
    iop, ax = AX[plane]
    for k in range(n):
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(d / f"{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Rows = ds.Columns = size; ds.PixelSpacing = [spacing, spacing]; ds.ImageOrientationPatient = iop
        pos = [0.0, 0.0, 0.0]; pos[ax] = 3.0 * k; ds.ImagePositionPatient = pos; ds.InstanceNumber = k + 1
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"; ds.BitsAllocated = 16; ds.BitsStored = 16
        ds.HighBit = 15; ds.PixelRepresentation = 0
        yy, xx = np.mgrid[:size, :size]; img = 800 * np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size / 5) ** 2))
        img = img * (0.5 + fluid * 0.5) + rng.normal(0, 20, img.shape); ds.PixelData = np.clip(img, 0, 4000).astype(np.uint16).tobytes()
        ds.save_as(str(d / f"{k}.dcm"), enforce_file_format=True)
    return sid

def main(out, n_train=36, n_test=6, n_gold=8, seed=0):
    out = pathlib.Path(out); rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("test", n_test)):
        studies = [f"1.2.826.{split}.{i}" for i in range(n)]; rows, ser = [], []
        for i, st in enumerate(studies):
            lab = (rng.random(12) < 0.3).astype(int)
            rep = ". ".join(PHRASE[l] if lab[j] else f"no {PHRASE[l]}" for j, l in enumerate(LABELS) if l in PHRASE) + "."
            row = {"StudyInstanceUID": st, "Report": rep}
            row.update({l: (int(lab[j]) if i < n_gold else np.nan) for j, l in enumerate(LABELS)}); rows.append(row)
            plans = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)] if i % 5 else [("Sagittal", 1), ("Coronal", 0)]
            if split == "test" and i == n - 1: plans = []                      # a test study with no series at all
            for plane, fl in plans:
                sid = write_series(out / f"{split}_series", st, plane, fl, rng, n=int(rng.integers(6, 11)))
                ser.append({"StudyInstanceUID": st, "SeriesInstanceUID": sid, "Anatomical_Plane": plane,
                            "Fluid_Sensitive": fl, "Fat_Suppression": fl})
        df = pd.DataFrame(rows)
        (df if split == "train" else df[["StudyInstanceUID"]]).to_csv(out / f"{split}.csv", index=False)
        pd.DataFrame(ser, columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).to_csv(out / f"{split}_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [f"1.2.826.test.{i}" for i in range(n_test)], **{l: 0.5 for l in LABELS}}).to_csv(out / "sample_submission.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1])
