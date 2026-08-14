# Building the manuscript locally

Overleaf is still the source of truth for drafting, but the paper builds
locally with [Tectonic](https://tectonic-typesetting.github.io/), a single
binary that downloads only the LaTeX packages this document needs. No MacTeX
install and no `sudo`.

## One-time setup

Homebrew works if your taps are trusted (`brew install tectonic`). Otherwise
grab the official release binary:

```sh
mkdir -p ~/.local/bin
URL=$(curl -sL https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
  | grep -o '"browser_download_url": *"[^"]*aarch64-apple-darwin[^"]*\.tar\.gz"' \
  | head -1 | sed 's/.*": *"//; s/"$//')
curl -sL "$URL" | tar xz -C ~/.local/bin tectonic
chmod +x ~/.local/bin/tectonic
```

Use `x86_64-apple-darwin` on an Intel Mac.

## Build

```sh
tectonic getfast-weather-modeling.tex
```

Tectonic runs the LaTeX and BibTeX passes to convergence on its own, so one
invocation is enough. The first build downloads packages and fonts and takes a
couple of minutes; later builds are fast. Add `-k` to keep `.aux`/`.xdv`
intermediates when debugging.

To confirm a build is actually clean rather than merely finishing:

```sh
tectonic getfast-weather-modeling.tex 2>&1 | grep -cE "undefined|multiply defined"
```

Zero is the expected answer.

## Verifying the physics numbers

Every number in the compensability appendix is reproduced by a self-checking
script that prints an ok/FAIL line per claim and a summary:

```sh
python3 verify_heat_convexity.py
```

It is stdlib-only and needs no arguments. The expected last line is
`ALL CHECKS PASSED`.

## Rebuilding the appendix figures

The three figures in the compensability appendix are generated from the same
constants as the verification script, so a change to the physics must be made
in both and checked in both:

```sh
python3 build_compensability_figures.py
```

This needs `matplotlib` and `numpy`, and writes vector PDFs straight into
`figures/`. The other figure scripts in the repo (`build_paper_figures.py`,
`build_headline_figures.py`, `build_lit_comparison.py`) read model artifacts
that live on the training box and will not run here.

## A portability note

TikZ node styles must not be named after built-in TikZ keys. The pipeline
figure originally defined a style called `out`, which collides with
`/tikz/out` (the outgoing-angle option on `to` paths). Overleaf's older PGF
tolerated it; newer PGF halts with "the key '/tikz/out' requires a value". It
is now `outbox`. Watch for the same trap with `in`, `at`, `text`, and `name`.
