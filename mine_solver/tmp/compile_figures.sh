latex --shell-escape circ.tex
rm circ.pdf circ.auxlock circ.aux circ.dvi circ.log circ-figure0.ps circ-figure0.log circ-figure0.dvi circ-figure0.dpth circ-figure0.md5
mv circ-figure0.eps circuit.eps
epstopdf circuit.eps
python figure.py
mv fig.eps hardware.eps

