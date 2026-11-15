# Report build

    cd reports/pdf
    python3 build_report.py body.pdf
    python3 "$PDF_SKILL_DIR/scripts/cover_render.py" 05 --out cover_05.pdf --content cover.json
    # then merge cover + body via pypdf and add metadata

Output: `MESHWEAVE-POC-Report.pdf` (5 pages).

Palette generated with `pdf.py palette.cascade`. Cover is template 05.
