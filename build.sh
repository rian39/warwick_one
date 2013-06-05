#!/bin/bash

./knit_all.sh
pandoc --smart --normalize --latex-engine=xelatex  --template=template.latex T--bibliography=references/refs.bib warwick.md -o warwick.pdf

evince warwick.pdf