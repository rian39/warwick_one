#!/bin/sh

pandoc --smart --normalize --latex-engine=xelatex  --template=template.latex --bibliography=references/refs.bib warwick.rmd -o warwick.pdf

evince warwick.pdf