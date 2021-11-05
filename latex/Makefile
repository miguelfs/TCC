SUBDIRS = . latex

make clean:
	rm -f 	rm -rf *.pdf *.abx *.aux *.bbl *.blg *.fdb_latexmk *.fls *.ilg \
	 *.lab *.lof *.log *.los *.out *.gz *. *.toc *.lot *syx*.DS_store *.ptb \
	 *.tod *.syx *.tdo
						 
make run:
	latexmk && pdflatex main && bibtex main && pdflatex main && pdflatex main