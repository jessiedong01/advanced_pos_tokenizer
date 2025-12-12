#!/usr/bin/env bash
# Build the arXiv source bundle: compiled .bbl, sections, tables, PDF figures and the bundled Tamil font.
set -euo pipefail
cd "$(dirname "$0")/../paper"
~/.local/bin/tectonic -X compile main.tex --keep-intermediates --keep-logs > /dev/null
rm -rf ../arxiv && mkdir -p ../arxiv/figures ../arxiv/fonts
cp main.tex numbers.tex refs.bib main.bbl ../arxiv/
cp -r sections tables ../arxiv/
cp figures/*.pdf ../arxiv/figures/
cp fonts/NotoSerifTamil-Regular.ttf fonts/OFL.txt ../arxiv/fonts/
printf '%%\\&xelatex\n' > ../arxiv/00README.XXX_engine_hint.txt
rm ../arxiv/00README.XXX_engine_hint.txt
cd ../arxiv && tar czf ../tokenizer-paper-arxiv.tar.gz . && cd .. && echo "bundle: $(tar tzf tokenizer-paper-arxiv.tar.gz | grep -vc '/$') files, $(du -h tokenizer-paper-arxiv.tar.gz | cut -f1)"
