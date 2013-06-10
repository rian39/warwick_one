#!/bin/bash

./knit_presentation.sh
pandoc warwick_presentation.md  -t html5  -o warwick_presentation.html
firefox warwick_presentation.html