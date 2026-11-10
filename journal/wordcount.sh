#!/bin/sh
# wordcount.sh - word counts for the title page (\wcAbstract, \wcMain in numbers.tex).
# Builds a copy without line numbers (\wordcountbuild; floats already follow the references), extracts its text with
# pdftotext (so number macros are expanded), and counts whitespace-separated words:
#   abstract  = from "Purpose:" up to "Keywords:"  (section labels included)
#   main text = from the "Introduction" heading up to the "Declarations" heading (headings included)
# Run from journal/: sh wordcount.sh
set -e
pdflatex -interaction=batchmode -jobname=wc "\def\wordcountbuild{1}\input{getfast-heat-journal}" >/dev/null 2>&1 || true
cp getfast-heat-journal.bbl wc.bbl 2>/dev/null || true
pdflatex -interaction=batchmode -jobname=wc "\def\wordcountbuild{1}\input{getfast-heat-journal}" >/dev/null 2>&1 || true
pdflatex -interaction=batchmode -jobname=wc "\def\wordcountbuild{1}\input{getfast-heat-journal}" >/dev/null 2>&1 || true
pdftotext -nopgbrk wc.pdf wc.txt
python3 - <<'PY'
import re
t = open("wc.txt").read()
t = re.sub(r"\n\s*\d+\s*\n", "\n", t)                                     # drop page numbers on their own line
a = t[t.index("Purpose:"):t.index("Keywords:")]
i0 = re.search(r"\n1\s*\n?\s*Introduction\s*\n", t).start()                 # the numbered heading, not a mention
i1 = [m_.start() for m_ in re.finditer(r"\nDeclarations\s*\n", t)][-1]            # the unnumbered heading
m = t[i0:i1]
cnt = lambda x: len(x.split())
# red draft markers are not prose: drop "OTn" superscripts and "[pending: ...]" brackets before counting
clean = lambda x: re.sub(r"\[pending:[^\]]*\]", " ", re.sub(r"\bOT\d+\b", " ", x))
print("abstract words:", cnt(a), "| without draft markers:", cnt(clean(a)))
print("main-text words:", cnt(m), "| without draft markers:", cnt(clean(m)))
PY
rm -f wc.aux wc.log wc.out wc.bbl wc.blg wc.pdf
