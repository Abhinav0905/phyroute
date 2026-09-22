"""Build and verify a compact reproducibility archive without publishing it."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PDF = OUTPUT / "pdf/PhyRoute_Revised_ICDLAI2026.pdf"
PREFIX = "phyroute-v0.2.0"
ARCHIVE = OUTPUT / f"{PREFIX}.zip"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(path):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(part in {".git", ".venv", "__pycache__", ".pytest_cache", "tmp"}
           or part.endswith(".egg-info") for part in parts):
        return True
    if path.name == ".DS_Store" or path.suffix == ".pyc":
        return True
    if str(rel).startswith(("paper/build/", "validation/runner_smoke/")):
        return True
    if parts[0] == "output" and path != PDF:
        return True
    if str(rel) in {"release_manifest.json", "paper/build_content.py",
                    "paper/make_docx.js", "scripts/run_all.py", "configs/default.yaml"}:
        return True
    if str(rel).startswith("paper/vendor/"):
        # Keep the complete supplied LaTeX package and its original archive.
        if "latex-package" not in parts and path.name != "latex-package.zip":
            return True
    return False


def regenerable_cache(path):
    rel = str(path.relative_to(ROOT))
    return path.suffix == ".npz" and any(rel.startswith(p) for p in (
        "results/revised/data/", "results/revised/physics/", "results/revised/predictions/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Refresh the current tree manifest without replacing any release ZIP.")
    args = parser.parse_args()
    if not PDF.is_file():
        raise SystemExit("Copy the inspected final PDF to output/pdf before packaging.")
    qa = json.loads((ROOT / "validation/pdf_visual_review.json").read_text())
    if not qa["passed"] or qa["pdf_sha256"] != digest(PDF):
        raise SystemExit("Visual review does not cover the current final PDF.")
    integrity = json.loads((ROOT / "validation/revised_integrity.json").read_text())
    if not integrity["passed"] or not integrity["complete"]:
        raise SystemExit("The revised experiment audit has not passed.")
    files, omitted = [], []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or excluded(path):
            continue
        entry = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                 "sha256": digest(path)}
        if regenerable_cache(path):
            omitted.append(entry)
        else:
            files.append(entry)
    manifest = {
        "version": "0.2.0", "status": "research software release; companion manuscript remains an author-review draft",
        "run_hash": json.loads((ROOT / "results/revised/run_manifest.json").read_text())["run_hash"],
        "included_files": files, "omitted_regenerable_npz": omitted,
        "omission_note": "Raw scenario, physics and prediction arrays remain in the full local workspace. "
                         "The archive retains their metadata and hashes. Regenerate with scripts/run_pipeline.py "
                         "before replaying the complete independent audit. Saved model arrays are included.",
        "manifest_note": "This manifest describes its accompanying tree. The manifest itself is "
                         "excluded from its own digest list. Published tags retain their original manifest.",
    }
    manifest_path = ROOT / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.manifest_only:
        print(json.dumps({"manifest": str(manifest_path), "included_files": len(files) + 1,
                          "omitted_cache_files": len(omitted), "archive_unchanged": True}, indent=2))
        return
    OUTPUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in files:
            archive.write(ROOT / entry["path"], f"{PREFIX}/{entry['path']}")
        archive.write(manifest_path, f"{PREFIX}/release_manifest.json")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise SystemExit("ZIP CRC verification failed.")
        if len(archive.namelist()) != len(files) + 1:
            raise SystemExit("Unexpected archive member count.")
        for entry in files:
            data = archive.read(f"{PREFIX}/{entry['path']}")
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise SystemExit(f"Archive digest mismatch: {entry['path']}")
    archive_sha = digest(ARCHIVE)
    ARCHIVE.with_suffix(".zip.sha256").write_text(f"{archive_sha}  {ARCHIVE.name}\n")
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
                      "sha256": archive_sha, "included_files": len(files) + 1,
                      "omitted_cache_files": len(omitted), "verified": True}, indent=2))


if __name__ == "__main__":
    main()
