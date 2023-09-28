


comp: report.pdf
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error report.tex
	bibtex report
        
clean:
	-rm -f *.aux
	-rm -f *.log
	-rm -f *.toc
	-rm -f *.bbl
	-rm -f *.blg
	-rm -f *.out
	-rm -f *.fls
	-rm -f *.fdb_latexmk
	-rm -f *.synctex.gz
	-rm -f *.xdv
	-rm -r -f _minted-report

