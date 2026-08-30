SHELL := /bin/sh
ROOT := ../../..
BUILD := $(ROOT)/tmp/pdfs/manuscript
OUTPUT := $(ROOT)/output/pdf/risk_does_not_specify_intervention_glass_scoped.pdf

.PHONY: pdf audit clean

pdf:
	mkdir -p $(BUILD) $(ROOT)/output/pdf
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD) manuscript.tex
	cp $(BUILD)/manuscript.pdf $(OUTPUT)

audit:
	python $(ROOT)/scripts/iclr27/audit_paper_package.py

clean:
	latexmk -C -outdir=$(BUILD) manuscript.tex
