"""Pack a paper folder for Overleaf: main.tex, refs.bib, the IEEEtran class and
style, and only the figures main.tex uses, into a clean folder and a zip, then
test-build the folder with latexmk so the upload is known to compile.

    python Tools/make_overleaf_zip.py docs/paper/exam2027 C:/Users/alexa/Downloads/EyeExam_ICCE_overleaf
"""
import os
import re
import shutil
import subprocess
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    tex = open(os.path.join(src, "main.tex"), encoding="utf-8").read()
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "figures"))
    for name in ("main.tex", "refs.bib", "IEEEtran.cls", "IEEEtran.bst"):
        shutil.copy(os.path.join(src, name), out)
    for f in sorted(used):
        shutil.copy(os.path.join(src, "figures", f), os.path.join(out, "figures", f))
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("Upload this folder (or the zip beside it) to Overleaf as a new project.\n"
                 "Main document: main.tex. Compiler: pdfLaTeX. Bibliography: BibTeX (refs.bib).\n"
                 "Figures are vector PDFs with embedded text in figures/.\n"
                 "IEEEtran.cls and IEEEtran.bst are included so the project builds without Overleaf's copies.\n")
    # test-build a copy, so the build files never land in the upload folder
    test = out + "_build"
    if os.path.isdir(test):
        shutil.rmtree(test)
    shutil.copytree(out, test)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=test,
                       capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(os.path.join(test, "main.pdf"))
    pages = None
    if ok:
        try:
            import fitz
            pages = len(fitz.open(os.path.join(test, "main.pdf")))
        except Exception:
            pass
    shutil.rmtree(test, ignore_errors=True)
    zip_path = shutil.make_archive(out, "zip", out)
    print(f"folder {out}\nzip    {zip_path}\nfigures {sorted(used)}\n"
          f"test build {'OK' if ok else 'FAILED'}" + (f", {pages} pages" if pages else ""))
    if not ok:
        print(r.stdout[-1500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
