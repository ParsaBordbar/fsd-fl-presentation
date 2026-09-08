# Federated Learning and the Future of Private, Open AI

Software Freedom Day talk — 28 reveal.js slides in the Canva deck's design (blue `#1e3a8a` / cream `#f5e8c7`, Anton + Montserrat). Slides 1–19 and 21–24 carry the Canva text; the "20 — An Example" placeholder is five slides of measured results from `demo/`. Each slide carries a `@minute · budget` stamp; the Q&amp;A slide lands at 33.5 min inside a 40-minute slot.

- `index.html` — the deck. Open in a browser; works fully offline (everything is in `vendor/`).
- `vendor/` — reveal.js 4.5.0, notes plugin, Chart.js 4.4.1, and the typefaces. `fonts-canva.css` (Anton, Montserrat, JetBrains Mono) is what the deck uses; `fonts.css` serves the legacy deck.
- `legacy/index-dark.html` — the previous dark, eight-colour version, kept for reference.

Keys: `←` `→` slides · `Esc` overview · `S` speaker notes · `F` fullscreen.

The demo tables and charts (FedAvg weights, accuracy per round, accuracy vs noise) are real numbers from `demo/`; regenerate with `python demo/fedavg_numpy.py`. Still to fill in before the talk: speaker name (slide 1).

## Single-file build and deploy

`python3 build_single.py` writes `dist/index.html` with reveal.js, the notes
plugin, Chart.js, and all four typefaces inlined (about 2 MB). That one file
is the whole deck.

`deploy/nginx.conf` is a server block that serves it. Install steps are at the
top of the file. Rebuild and re-copy after every edit to `index.html`.

The demo code the "Live demo" slide describes lives in `demo/` (package
`demo/hospitals_fl/`); see `demo/README.md` for the commands and expected numbers.
