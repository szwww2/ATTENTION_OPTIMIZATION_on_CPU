#!/usr/bin/env bash
# Rebuild the report PDF from main.tex and team_inputs.tex.
#   ./build.sh            compile the PDF (pdflatex twice)
#   ./build.sh --numbers  first regenerate generated/ and figures/ from the archived data
#                         (needs this folder to be <repo>/report, plus NumPy and Matplotlib)
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--numbers" ]; then
    "${PYTHON:-python3}" make_numbers.py
fi
for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
done
cp main.pdf GP8_Final_Report.pdf
echo "Wrote GP8_Final_Report.pdf ($(grep -c 'TO FILL' <(pdftotext main.pdf - 2>/dev/null) || true) placeholder(s) left)"
