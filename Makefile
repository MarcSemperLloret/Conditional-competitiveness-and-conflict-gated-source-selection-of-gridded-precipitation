.PHONY: all tables figures

all:
	python scripts/regenerate_manuscript_outputs.py

tables:
	python scripts/regenerate_manuscript_outputs.py

figures:
	python scripts/regenerate_manuscript_outputs.py
