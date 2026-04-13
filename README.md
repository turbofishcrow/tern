A microtonal scale research app chiefly dedicated to ternary scales (scales with three distinct step sizes). It focuses on [aberrismic theory](https://xenreference.com/w/Aberrisma), developed by groundfault, inthar, and others.

![Front page screenshot](https://raw.githubusercontent.com/turbofishcrow/ternary/main/static/images/front.png)

# How to build and run

## Prerequisites

- [Rust](https://rustup.rs/) (for compiling to WebAssembly)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/) (v18+ recommended)

## Build steps

1. Install Node.js dependencies:

   ```bash
   npm install
   ```

2. Build and serve the development server:

   ```bash
   npm run serve
   ```

   This will compile the Rust code to WASM and start a dev server at `http://localhost:8080/`.

3. For a production build:
   ```bash
   npm run build
   ```
   The output will be in the `dist/` directory.

## Rust development

To check the Rust code compiles:

```bash
cargo check
```

To run Rust tests:

```bash
cargo test
```

# Features

- Get the set of all scales (up to mode) with a certain step signature.
- Given a step signature, it gives you tuples of JI steps with bounded complexity for the scale, assuming octave equivalence. (Technical note: A list of 270edo interval detemperings are used for the fast solver. The slow solver (accessed by "Get more JI tunings" button) uses 27-odd-limit intervals as targets it tries to match scale intervals to.)
- Given step signature, it displays the ed(equave) tunings.
- When you select a tuning on the results page, the SonicWeave code is displayed.
- JI-agnostic 2D lattice view for every scale.
- Non-octave equaves are supported (enter as a JI ratio like "3/1").
- Configurable tuning bounds:
  - Max ED size (default 111)
  - Min/max smallest step size in cents (default 20–250)
- Every scale comes with a Scale Profile that shows properties of the scale selected or queried
  - guide frame (guided generator sequence; multiplicity or interleaving offset_chord; complexity)
  - monotone MOS properties satisfied (L=M, M=s, s=0)
  - maximum variety
- Filter for
  - whether the scale is a MOS substitution scale
  - length of the guided generator sequence
  - guide frame complexity
  - monotone MOS properties
  - maximum variety
- URL parameters. Example URLs you can now use:
  - ?word=LLmLLms: Query scale word "LLmLLms"
  - ?word=LLmLLms&equave=3/1: Same with tritave equave
  - ?sig=3+2+2: Query step signature 3L2m2s
  - ?sig=5+2+2&type=all-scales: All scales with 5L2m2s
  - ?sig=3+2+2&s0=0: Disable s=0 filter
  - ?sig=3+2+2&mv=3&mvmode=at-most: Filter MV ≤ 3

# Nota bene

If you ever see the status message "RuntimeError: unreachable executed" while running the web app, it's a bug (something that's not supposed to happen is happening). Please report it. When an error happens, just refresh the web app.
