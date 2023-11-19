echo "Generating results"

python3 install_scripts/process_for_latex.py results latex/artifact_data


echo "Generating PDF of results"
cd latex
#needs two passes because of label names.
pdflatex -jobname=artifact -draftmode main.tex
pdflatex -jobname=artifact main.tex
cp artifact.pdf ../artifact.pdf
cd ..