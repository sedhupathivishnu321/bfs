"""Builds notebooks/export_backbone_weights.ipynb: a tiny Kaggle notebook (Internet ON, no GPU needed) that downloads the
pretrained backbone weights once and saves them as safetensors, so the main notebook can run with Internet OFF.

    python notebooks/build_export_weights_notebook.py
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = [
    md(r"""# Export pretrained backbone weights (run once, Internet ON)

1. In the right-hand **Settings**, turn **Internet On**. A GPU is not needed.
2. **Run All.** It downloads each backbone and writes `/kaggle/working/backbone_weights/<name>.safetensors`.
3. Click **Save Version** (Save & Run All). When it finishes, open the version, go to **Output**, and choose **New Dataset**
   (name it e.g. `knee-backbone-weights`).
4. In the real notebook: **Add Input** -> your dataset, turn **Internet Off**. The notebook finds the files by name.

The list below must match `CFG.BACKBONES` of the notebook you will run (`ultra` preset = the two defaults shown)."""),
    code(r"""import os, torch, timm
from safetensors.torch import save_file
BACKBONES = ["coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "convnext_small.in12k_ft_in1k"]   # add "convnext_tiny.in12k_ft_in1k" for PRESET="balanced"
IMG = 384                                  # the main notebook's CFG.IMG (only the hybrid CoAtNet/ViT families care)
OUT = "/kaggle/working/backbone_weights"; os.makedirs(OUT, exist_ok=True)
for b in BACKBONES:
    kw = {"img_size": IMG} if b.startswith(("coatnet", "coat_", "maxvit", "maxxvit", "vit_", "eva", "beit")) else {}
    m = timm.create_model(b, pretrained=True, num_classes=0, **kw)
    path = f"{OUT}/{b}.safetensors"
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, path)
    print(f"saved {path}  {os.path.getsize(path) / 1e6:.0f} MB  {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M params")
print(os.listdir(OUT))"""),
]
nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["kaggle"] = {"accelerator": "none", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/export_backbone_weights.ipynb"
nbf.write(nb, out)
print("wrote", out)
