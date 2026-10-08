# Paper

ACL template (`acl.sty`, `acl_natbib.bst`, `acl_latex.tex` from github.com/acl-org/acl-style-files).

    python analysis/figures.py          # from the repo root: figures/ and generated/ from results/
    cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

`analysis/figures.py` uses the full GPU run when its files are in `results/`
(after `results/ingest.py`), otherwise the CPU pilot; captions say which. Every number
in the text comes from `generated/numbers.tex`. Red `[TODO-RESULT: ...]` marks text to
write once the full run is in. `references.bib` lists only entries checked against the
publisher or arXiv record (evidence in `citation_checks/`).
