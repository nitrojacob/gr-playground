import json

nb_path = "/mnt/wksp/kaggle_5dag/experiments/gr-playground/benchmarks/radioml2018-mlamc-results.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    if cell["cell_type"] == "code":
        for out in cell.get("outputs", []):
            if "text" in out:
                print("STDOUT/STDERR:")
                print("".join(out["text"])[:500])
            elif "data" in out:
                print("DATA DISPLAYED:", list(out["data"].keys()))
