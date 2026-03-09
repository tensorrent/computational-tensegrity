.PHONY: all data figures paper clean

all: data figures paper

data:
	python scripts/compute_all.py

figures:
	python scripts/gen_extended_figs.py

paper:
	cd paper && \
	cp ../figures/*.pdf . && \
	cp ../src/zeta.py ../rc_zeta_listing.py && \
	cp ../src/sigma_engine.py ../rc_sigma_listing.py && \
	cp ../src/main.py ../rc_main_listing.py && \
	pdflatex Computational_Tensegrity_Wallace_2026.tex && \
	pdflatex Computational_Tensegrity_Wallace_2026.tex && \
	pdflatex Computational_Tensegrity_Wallace_2026.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.bbl paper/*.blg paper/*_listing.py paper/*.pdf
	rm -f paper/Computational_Tensegrity_Wallace_2026.pdf
	rm -f figures/*.pdf
	rm -f scripts/*.json
