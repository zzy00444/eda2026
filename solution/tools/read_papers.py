"""Extract local papers for algorithm review, preserving page boundaries."""
from pathlib import Path
import fitz

root = Path(__file__).resolve().parents[2]
out = root / "solution/build/paper_text"
out.mkdir(parents=True, exist_ok=True)
for file in (root / "paper").glob("*.pdf"):
    doc = fitz.open(file)
    text = "\n".join(f"\n--- Page {i+1} ---\n{p.get_text()}" for i, p in enumerate(doc))
    (out / (file.stem + ".txt")).write_text(text, encoding="utf-8")
    print(file.name, len(doc), doc.metadata, flush=True)
    print(text[:7000], flush=True)
    page = doc[0]
    page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(out / (file.stem + "_page1.png"))
