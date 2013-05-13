#!/bin/sh

 pandoc  --smart --normalize --latex-engine=xelatex  --template=template.latex --bibliography=references/refs.bib warwick_paper.rmd -o warwick_paper.pdf 
 # pandoc  --smart --normalize --latex-engine=xelatex   --bibliography=references/refs.bib warwick_paper.rmd -o warwick_paper.pdf 
 evince warwick_paper.pdf