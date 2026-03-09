# Computational Tensegrity

**Asymmetric Coupling, Phase Boundaries, and Catastrophic Failure in Nonlinear Codependent Systems**

Brad Wallace · Independent Researcher · [TensorRent](https://github.com/tensorrent)

---

## Overview

Every coupled nonlinear system is a tensegrity structure: cables (restoring forces, βκ) and struts (destabilising loads, αγ) in asymmetric codependence. A single scalar ratio ρ = αγ/βκ governs the phase boundary. When ρ < 1, the structure holds. When ρ crosses 1, collapse is catastrophic—not gradual.

This repository contains the paper, source code, computed figures, and reproducibility scripts for the complete framework, validated across six domains:

1. **IEEE-754 float error propagation** — per-operation tolerances compound through nonlinear chains to cross decision thresholds
2. **Pure-integer prime wave computation** — 1,000/1,000 exact match with double precision using Q40 fixed-point
3. **BRA integer kernel** — 1.6×10¹¹ times more precise than TensorFlow f32 at 14 KB
4. **Quantum circuits** — 35.4% Toffoli gate reduction via integer CORDIC
5. **Wolfram hypergraph rewriting** — novel tensegrity ratio ρ(A)/λ₂ reveals spectral fragility; spectral causal invariance fails
6. **Transparent Consent governance** — consensus as lossy compression; constraint satisfaction preserves informational dimensionality

## Key Results

| Result | Value |
|--------|-------|
| Universal stability condition | ρ = αγ/βκ < 1 |
| Mode collapse law | β_c·a² = (8ω/3Γ)·Δω (confirmed ±6%) |
| Epistemic horizon | σ_c = C·A·λ^α·N^(-β/D₂) |
| BRA vs TF f32 precision | 1.6 × 10¹¹ × |
| Integer vs float Toffoli | 35.4% reduction |
| Wolfram tensegrity ratio | Up to T = 37.3 (fragile) |
| Spectral causal invariance | Fails (λ₂ spread up to 0.48) |

## Repository Structure

```
paper/          Compiled PDF and LaTeX source
figures/        All 14 computed figures (PDF)
src/            RC Epistemic Constraint Engine (Python)
  zeta.py         ζ(S) invariant anchor — five structural gates
  sigma_engine.py Σ-Engine — spectral proximity instrument
  main.py         Turn-key demonstration
scripts/        Reproducibility
  compute_all.py        Prime wave, resonance scores, stability experiments
  gen_extended_figs.py  All figure generation
  wolfram_rc_eval.py    Wolfram hypergraph evaluation with causal invariance test
  results.json          Computed data (prime wave)
  wolfram_data.json     Computed data (Wolfram spectral evolution)
```

## Quick Start

```bash
pip install numpy scipy mpmath matplotlib
python src/main.py              # RC stack demo
python scripts/compute_all.py   # Regenerate all prime wave data and figures
python scripts/wolfram_rc_eval.py  # Wolfram hypergraph evaluation
```

## Compiling the Paper

```bash
cd paper
# Copy figures alongside .tex
cp ../figures/*.pdf .
# Also copy code listings for appendix
cp ../src/zeta.py ../rc_zeta_listing.py
cp ../src/sigma_engine.py ../rc_sigma_listing.py
cp ../src/main.py ../rc_main_listing.py
pdflatex Computational_Tensegrity_Wallace_2026.tex
pdflatex Computational_Tensegrity_Wallace_2026.tex
pdflatex Computational_Tensegrity_Wallace_2026.tex
```

## The Octagon Problem

Eight observers view one face of an octagonal solid. Each observation is locally correct. Each conclusion about the whole object is globally incorrect. The failure is topological: the communication graph has zero edges. This is the universal failure mode formalized in the paper.

**Truth is not hidden. It is distributed.**

## License

**Sovereign Integrity Protocol License (SIP v1.1)**

- **Personal, educational, and individual use**: Free. Always. In perpetuity. Irrevocable.
- **Commercial use**: Requires written license from the author. Default 5% of gross revenue in perpetuity.
- **Unlicensed commercial use**: Automatic 4.2% of gross revenue to the author + 4.2% to public infrastructure (8.4% total). Non-negotiable. Survives all transfers. Binds downstream users. In perpetuity.

See [LICENSE](LICENSE) for the full text.

## Contact

Brad Wallace · Wellthatshandy@gmail.com · [github.com/tensorrent](https://github.com/tensorrent)

## Acknowledgements

4,500 hours over twelve months. AI assistants (Anthropic Claude, OpenAI GPT, Google Gemini, DeepSeek, Cursor AI, Antigravity) used in MOE cross-validation workflow. Over 60% of input via voice-to-text. Author has dyslexia and ADHD. All mathematical formulations, architecture, and research direction originated with the author.

This programme originated February 24, 2025.
