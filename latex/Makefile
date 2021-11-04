SUBDIRS = . latex

make clean:
	rm -f *.log *.aux *.idx *.blg *.bbl *.glo *.bz2 \
		         *.toc *.lof *.lot *.syx *.abx *.lab *.ilg *.los *.ind \
						 *.gls *.out *~ coppe.pdf example.pdf \
						 *.fdb_latexmk *.fls *.gz \
						 coppe-logo-eps-converted-to.pdf \
						 poli-logo-eps-converted-to.pdf \
						 *.dvi *.DS_store *.ptb *.tod
						 
make run:
	latexmk && pdflatex main && bibtex main && pdflatex main && pdflatex main