"""
Builds data/plantvillage and data/plantdoc by downloading PlantVillage
(Hughes and Salathe, 2015) and PlantDoc (Singh et al., 2020) from their
official GitHub repositories and copying the 16 shared classes used in the
paper into a common folder layout (data/<dataset>/<class_name>/).

Requires: git installed, about 2GB free disk, internet access.

Run from the repository root: python scripts/download_real_data.py
"""

import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(REPO_ROOT, "data")
TMP_DIR = os.path.join(REPO_ROOT, ".tmp_download")

# The 16 classes present in all three datasets (see Section 3.1 of the paper).
# Two additional classes from an initial candidate set were excluded: a
# tomato two-spotted spider mite class (only 2 PlantDoc images), and a bell
# pepper bacterial spot class (PlantWild's matching label, "bell pepper leaf
# spot", does not specify bacterial etiology and was not visually confirmed
# as an exact match). To add the latter back after a visual check, uncomment
# the line below and add "bell pepper leaf spot" to PLANTWILD_MAP in
# integrate_plantwild.py.
SHARED_CLASSES = {
    "apple_scab": ("Apple___Apple_scab", "Apple Scab Leaf"),
    "apple_rust": ("Apple___Cedar_apple_rust", "Apple rust leaf"),
    "corn_gray_leaf_spot": ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
    "corn_common_rust": ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
    "corn_northern_leaf_blight": ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
    "grape_black_rot": ("Grape___Black_rot", "grape leaf black rot"),
    # "pepper_bell_bacterial_spot": ("Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot"),  # excluded, see note above
    "potato_early_blight": ("Potato___Early_blight", "Potato leaf early blight"),
    "potato_late_blight": ("Potato___Late_blight", "Potato leaf late blight"),
    "squash_powdery_mildew": ("Squash___Powdery_mildew", "Squash Powdery mildew leaf"),
    "tomato_bacterial_spot": ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    "tomato_early_blight": ("Tomato___Early_blight", "Tomato Early blight leaf"),
    "tomato_late_blight": ("Tomato___Late_blight", "Tomato leaf late blight"),
    "tomato_leaf_mold": ("Tomato___Leaf_Mold", "Tomato mold leaf"),
    "tomato_septoria_leaf_spot": ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    "tomato_yellow_leaf_curl_virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    "tomato_mosaic_virus": ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
}


def run(cmd, **kwargs):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def harmonize(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- PlantDoc (small, ~1.9GB clone but working tree is ~550MB) ---
    plantdoc_src = os.path.join(TMP_DIR, "PlantDoc-Dataset")
    if not os.path.isdir(plantdoc_src):
        print("\n=== Cloning PlantDoc-Dataset ===")
        run(f"git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git {plantdoc_src}")
        shutil.rmtree(os.path.join(plantdoc_src, ".git"), ignore_errors=True)

    # --- PlantVillage (sparse checkout of raw/color only, ~900MB) ---
    plantvillage_src = os.path.join(TMP_DIR, "PlantVillage-Dataset")
    if not os.path.isdir(os.path.join(plantvillage_src, "raw", "color")):
        print("\n=== Cloning PlantVillage-Dataset (raw/color only) ===")
        run(f"git clone --filter=blob:none --sparse --depth 1 "
            f"https://github.com/spMohanty/PlantVillage-Dataset.git {plantvillage_src}")
        run("git sparse-checkout set raw/color", cwd=plantvillage_src)

    pv_color_dir = os.path.join(plantvillage_src, "raw", "color")
    pd_train_dir = os.path.join(plantdoc_src, "train")
    pd_test_dir = os.path.join(plantdoc_src, "test")

    pv_dest = os.path.join(DATA_DIR, "plantvillage")
    os.makedirs(pv_dest, exist_ok=True)
    pd_dest = os.path.join(DATA_DIR, "plantdoc")
    os.makedirs(pd_dest, exist_ok=True)

    # Build case-insensitive lookup of PlantDoc's actual raw folder names,
    # since train/ and test/ can have slightly inconsistent casing/spacing.
    def build_lookup(split_dir):
        if not os.path.isdir(split_dir):
            return {}
        return {name.lower(): name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name))}

    pd_train_lookup = build_lookup(pd_train_dir)
    pd_test_lookup = build_lookup(pd_test_dir)

    print("\n=== Copying shared classes into data/plantvillage and data/plantdoc ===")
    pv_counts, pd_counts = {}, {}
    for canonical, (pv_folder, pd_folder_query) in SHARED_CLASSES.items():
        # PlantVillage
        src = os.path.join(pv_color_dir, pv_folder)
        dest = os.path.join(pv_dest, canonical)
        os.makedirs(dest, exist_ok=True)
        files = os.listdir(src)
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        pv_counts[canonical] = len(files)

        # PlantDoc (merge train + test, case-insensitive folder match)
        dest2 = os.path.join(pd_dest, canonical)
        os.makedirs(dest2, exist_ok=True)
        pd_count = 0
        query_lower = pd_folder_query.lower()
        for lookup, split_dir, split_tag in [(pd_train_lookup, pd_train_dir, "train"), (pd_test_lookup, pd_test_dir, "test")]:
            actual_folder = lookup.get(query_lower)
            if actual_folder is None:
                continue
            src2 = os.path.join(split_dir, actual_folder)
            for f in os.listdir(src2):
                shutil.copy2(os.path.join(src2, f), os.path.join(dest2, f"{split_tag}_{f}"))
                pd_count += 1
        pd_counts[canonical] = pd_count

    print(f"\n{'Class':<32} {'PlantVillage':>13} {'PlantDoc':>10}")
    for c in SHARED_CLASSES:
        print(f"{c:<32} {pv_counts[c]:>13} {pd_counts[c]:>10}")
    print(f"\nTotal: PlantVillage={sum(pv_counts.values())} images, "
          f"PlantDoc={sum(pd_counts.values())} images, {len(SHARED_CLASSES)} shared classes")

    print("\nCleaning up temporary clones...")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Done. data/plantvillage and data/plantdoc are ready.")


if __name__ == "__main__":
    main()
