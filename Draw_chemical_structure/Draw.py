from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# --- Folder of the current script ---
script_dir = Path(__file__).resolve().parent

# --- Molecules and SMILES ---
molecules = {
    "DMF": "CN(C)C=O",  # N,N-Dimethylformamide
    "DMA": "CN(C)",     # Dimethylamine
    "MMA": "CN"         # Methylamine
}

for name, smiles in molecules.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error parsing SMILES for {name}: {smiles}")
        continue

    mol_h = Chem.AddHs(mol)
    rdDepictor.Compute2DCoords(mol_h)

    # --- PNG output ---
    png_path = script_dir / f"{name}.png"
    img = Draw.MolToImage(mol_h, size=(400, 200))
    img.save(png_path)
    print(f"Saved PNG: {png_path}")

    # --- SVG output ---
    svg_path = script_dir / f"{name}.svg"
    drawer = rdMolDraw2D.MolDraw2DSVG(450, 180)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_h)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved SVG: {svg_path}")

    # --- Molecular formula ---
    formula = rdMolDescriptors.CalcMolFormula(mol)
    print(f"{name} - SMILES: {smiles}, Formula: {formula}\n")
