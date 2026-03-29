// consts for the lattice view
const LATTICE_SVG_WIDTH = 500;
const LATTICE_SVG_HEIGHT = 500;
const ORIGIN_X = LATTICE_SVG_WIDTH / 2;
const ORIGIN_Y = LATTICE_SVG_HEIGHT / 2;
const SPACING_X = 35;
const SPACING_Y = -35;
const UNOCCUPIED_DOT_RADIUS = 7;
const EDGE_WIDTH = 2;

const GROUND_INDIGO = "#76f";

// Default bounds for tuning search
const DEFAULT_ED_BOUND = 111;
const DEFAULT_S_LOWER_BOUND = 20.0;
const DEFAULT_S_UPPER_BOUND = 250.0;

const statusElement = document.getElementById("status");

/**
 * Count occurrences of a character in a string
 */
function countChar(str, char) {
  return [...str].filter((c) => c === char).length;
}
function displayStepVector(vector) {
  const keys = Object.keys(vector);
  if (keys.length === 0) return "0";
  const sizeIdentifiers = ["L", "m", "s"];
  return keys.map((k, i) => `${vector[k]}${sizeIdentifiers[i]}`).join("+");
}

/**
 * Greatest Common Divisor using Euclidean algorithm
 */
function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b !== 0) {
    [a, b] = [b, a % b];
  }
  return a;
}

/**
 * Greatest Common Divisor for BigInt values
 */
function gcdBigInt(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b !== 0n) {
    [a, b] = [b, a % b];
  }
  return a;
}

/**
 * Python-style modulo (always positive result)
 */
function mod(n, m) {
  return ((n % m) + m) % m;
}

// Primes used in monzo representation
const MONZO_PRIMES = [2, 3, 5, 7, 11, 13];

/**
 * Parse a monzo string like "[-3, 2, 0, 0, 0, 0>" and return its value in cents
 * Returns null if parsing fails
 */
function monzoToCents(monzoStr) {
  // Match the monzo format: [exp, exp, ...>
  const match = monzoStr.trim().match(/^\[([^\]>]*)[>\]]?$/);
  if (!match) return null;

  const exponents = match[1].split(",").map((s) => Number(s.trim()));
  if (exponents.some(isNaN)) return null;

  // cents = 1200 * log2(product of prime^exponent)
  let cents = 0;
  for (let i = 0; i < exponents.length && i < MONZO_PRIMES.length; i++) {
    cents += exponents[i] * 1200 * Math.log2(MONZO_PRIMES[i]);
  }
  return cents;
}

/**
 * Parse an equave ratio string (e.g., "2/1") and convert to cents
 * Returns 1200 (octave) if parsing fails
 */
function parseEquaveToCents(ratioStr) {
  const match = ratioStr.trim().match(/^(\d+)\/(\d+)$/);
  if (!match) {
    return 1200; // Default to octave
  }
  const numerator = parseInt(match[1], 10);
  const denominator = parseInt(match[2], 10);
  if (denominator === 0 || numerator <= 0 || denominator <= 0) {
    return 1200; // Default to octave
  }
  // cents = 1200 * log2(ratio)
  return 1200 * Math.log2(numerator / denominator);
}

/**
 * Get the current equave ratio string from the input field (normalized)
 * Returns { ratio: "m/n", num: m, den: n }
 */
function getEquaveRatio() {
  const input = document.getElementById("input-equave");
  if (!input) {
    return { ratio: "2/1", num: 2, den: 1 };
  }
  const value = input.value.trim();
  const match = value.match(/^(\d+)\/(\d+)$/);
  if (!match) {
    return { ratio: "2/1", num: 2, den: 1 };
  }
  const num = parseInt(match[1], 10);
  const den = parseInt(match[2], 10);
  const d = gcd(num, den);
  return { ratio: `${num / d}/${den / d}`, num: num / d, den: den / d };
}

/**
 * Get the current equave in cents from the input field
 */
function getEquaveCents() {
  const input = document.getElementById("input-equave");
  return input ? parseEquaveToCents(input.value) : 1200;
}

/**
 * Get the ED bound from the input field
 */
function getEdBound() {
  const input = document.getElementById("input-ed-bound");
  return input
    ? parseInt(input.value, 10) || DEFAULT_ED_BOUND
    : DEFAULT_ED_BOUND;
}

/**
 * Get the minimum s size in cents from the input field
 */
function getSLower() {
  const input = document.getElementById("input-s-lower");
  return input
    ? parseFloat(input.value) || DEFAULT_S_LOWER_BOUND
    : DEFAULT_S_LOWER_BOUND;
}

/**
 * Get the maximum s size in cents from the input field
 */
function getSUpper() {
  const input = document.getElementById("input-s-upper");
  return input
    ? parseFloat(input.value) || DEFAULT_S_UPPER_BOUND
    : DEFAULT_S_UPPER_BOUND;
}

function stepVectorLength(vector) {
  return Object.values(vector).reduce((sum, v) => sum + v, 0);
}

// Mutates `arr` a flat array matrix, swapping rows `i1` and `i2`.
function swapRows(arr, m, n, i1, i2) {
  if (i1 < 0 || i1 >= m || i2 < 0 || i2 >= m) {
    throw new Error(`swapRows(): matrix index out of bounds!`);
  }
  for (let j = 0; j < n; j++) {
    [arr[n * i1 + j], arr[n * i2 + j]] = [arr[n * i2 + j], arr[n * i1 + j]];
  }
}

// Scales row `i` of a matrix by `coeff`.
function multiplyRow(arr, m, n, i, coeff) {
  for (let j = 0; j < n; j++) {
    arr[n * i + j] *= coeff;
  }
}

// Does the operation "[row i2 of arr] += coeff * [row i1 of arr]".
function addMultipleOfFirstRowToSecond(arr, _, n, i1, i2, coeff) {
  for (let j = 0; j < n; j++) {
    arr[n * i2 + j] += coeff * arr[n * i1 + j];
  }
}

// Makes a table in `tableElement` with the given `data`.
function makeTable(tableElement, data, header = "") {
  const tableViewTr = document.createElement("tr");
  let tableView = tableContent(data, header);
  tableViewTr.appendChild(tableView);
  tableElement.appendChild(tableViewTr);
}

// Return a new table view
function tableContent(data, header = "") {
  const table = tableHead(data, header);
  const tbody = table.createTBody();
  if (data[0] instanceof Array) {
    for (const [i] of data.entries()) {
      let row = tbody.insertRow();
      let cell1 = row.insertCell();
      cell1.appendChild(document.createTextNode(`${i + 1}`)); // row numbering
      for (const value of data[i].values()) {
        // iterate over columns
        let td = document.createElement("td");
        td.appendChild(document.createTextNode(value));
        row.appendChild(td);
      }
    }
  } else if (typeof data[0] === "object") {
    for (let i = 0; i < data.length; i++) {
      let row = tbody.insertRow();
      let cell1 = row.insertCell();
      cell1.appendChild(document.createTextNode(`${i + 1}`)); // row numbering
      for (const value of Object.values(data[i])) {
        // iterate over columns
        let td = document.createElement("td");
        td.appendChild(document.createTextNode(`${value}`));
        row.appendChild(td);
      }
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      let row = tbody.insertRow();
      let cell1 = row.insertCell();
      cell1.appendChild(document.createTextNode(`${i + 1}`)); //row numbering
      let td = document.createElement("td");
      td.appendChild(document.createTextNode(data[i]));
      row.appendChild(td);
    }
  }
  return table;
}

function tableHead(data, header = "") {
  const table = document.createElement("table");
  const thead = table.createTHead();
  table.setAttribute("class", "scrollable clickable");
  let headRow = thead.insertRow();
  let th = document.createElement("th");
  th.appendChild(document.createTextNode("#"));
  headRow.appendChild(th);
  if (data) {
    if (data[0] instanceof Array) {
      for (let i = 0; i < data[0].length; ++i) {
        let th = document.createElement("th");
        th.appendChild(document.createTextNode(`${i}`));
        headRow.appendChild(th);
      }
    } else if (typeof data[0] === "object") {
      for (let key of Object.keys(data[0])) {
        let th = document.createElement("th");
        th.appendChild(document.createTextNode(`${key}`));
        headRow.appendChild(th);
      }
    } else {
      let th = document.createElement("th");
      th.appendChild(document.createTextNode(`${header}`));
      headRow.appendChild(th);
    }
  }
  return table;
}

// Main application code - load WASM module
import("./pkg")
  .then((wasm) => {
    // app state
    const appState = {
      word: null,
      latticeBasis: null,
      tuning: null,
      profile: null,
    };

    // New approach:
    // 1. draw a background 2D grid first
    // 2. represent x and y directions as generator and offset, whichever way fits better on the screen
    // 3. choose a zero point
    // TODO: Indicate what kind of guide frame it is. (simple, multiple/interleaved)
    //
    // TODO: show a legend for the different colored lines
    function createLatticeView(state, equave) {
      if (statusElement) {
        statusElement.innerText = "";

        let A;
        let B;
        let C;
        let D;

        // Create lattice visualization
        const latticeElement = document.getElementById("lattice-vis");
        if (latticeElement) {
          latticeElement.innerHTML = "";
          latticeElement.setAttribute("style", "vertical-align: text-top;");
          const svgTag = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "svg",
          );
          svgTag.setAttribute("id", "lattice");
          svgTag.setAttribute("width", "100%");
          svgTag.setAttribute("height", "400");
          svgTag.style.touchAction = "none";
          svgTag.style.cursor = "grab";
          const svgStyle = document.createElement("style");
          svgStyle.innerHTML = `
      .small {
        font: 20px sans-serif;
        fill: black;
      }`;
          if (state.word) {
            if (state.latticeBasis) {
              let n = state.word.length;
              [A, B, C, D] = [1, 0, 0, 1];

              // Draw coordinate grid
              for (let i = -64; i < 64; ++i) {
                // Lines of constant x
                const p0x = ORIGIN_X + A * SPACING_X * i; // ith x offset
                const p0y = ORIGIN_X + B * SPACING_Y * i;
                const p1x = p0x + C * SPACING_X * 100; // add a large positive x offset
                const p1y = p0y + D * SPACING_Y * 100;
                const p2x = p0x + C * SPACING_X * -100; // add a large negative x offset
                const p2y = p0y + D * SPACING_Y * -100;
                // Lines of constant y
                const q0x = ORIGIN_X + C * SPACING_X * i;
                const q0y = ORIGIN_Y + D * SPACING_Y * i;
                const q1x = q0x + A * SPACING_X * 100;
                const q1y = q0y + B * SPACING_Y * 100;
                const q2x = q0x + A * SPACING_X * -100;
                const q2y = q0y + B * SPACING_Y * -100;

                // draw the line of constant x
                svgTag.innerHTML += `<line
                x1="${p1x}"
                y1="${p1y}"
                x2="${p2x}"
                y2="${p2y}"
                style="stroke:${GROUND_INDIGO}; stroke-width:${EDGE_WIDTH}"
              />`;
                // draw the line of constant y
                svgTag.innerHTML += `<line
                x1="${q1x}"
                y1="${q1y}"
                x2="${q2x}"
                y2="${q2y}"
                style="stroke:gray; stroke-width:${EDGE_WIDTH}"
              />`;
              }

              // Track cumulative step counts for pitch calculation
              let stepCounts = { L: 0, m: 0, s: 0 };
              // Get lattice coordinates and basis directly from WASM
              const latticeResult = wasm.word_to_lattice(state.word);
              if (latticeResult && latticeResult.basis) {
                // Update the state with the better basis
                state.latticeBasis = latticeResult.basis;
              }
              const latticeCoords = latticeResult.coordinates;
              for (let deg = 0; deg < latticeCoords.length; ++deg) {
                // Get coordinates directly from WASM-computed lattice
                const [latticeX, latticeY] = latticeCoords[deg];
                const currentX = ORIGIN_X + latticeX * SPACING_X;
                const currentY = ORIGIN_Y + latticeY * SPACING_Y;

                const { pitch, cents } = getPitchInfo(
                  stepCounts,
                  state.tuning,
                  equave,
                );
                const centsRounded = Math.round(cents);
                // If pitch already shows cents, don't duplicate
                const tooltipText = pitch.endsWith("¢")
                  ? `Degree ${deg}: ${pitch}`
                  : `Degree ${deg}: ${pitch} (${centsRounded}¢)`;

                svgTag.innerHTML += `<g class="note-point" style="cursor: pointer;">
            <circle
              cx="${currentX}"
              cy="${currentY}"
              r="${UNOCCUPIED_DOT_RADIUS}"
              fill="white"
              stroke="white"
              stroke-width="1"
            >
              <title>${tooltipText}</title>
            </circle>
            <text
              x="${currentX - 3}"
              y="${currentY + 3}"
              fill="black"
              font-size="0.5em"
              style="pointer-events: none;"
            >${mod(deg, n)}</text>
          </g>`;

                // Update step counts for next iteration
                stepCounts[state.word[deg]]++;
              }
              // We deferred appending elements until now
              // Initial viewBox will be set by updateViewBox() below
              latticeElement.innerHTML += `<hr/><h2>Lattice view</h2><br/><small>Ternary scales are special in that they admit a JI-agnostic 2D lattice representation.</small>`;
              latticeElement.innerHTML += `<br/><small>Hover over the dots to see pitch information. Click and drag to pan, use mouse wheel or buttons to zoom.</small>`;
              // Add zoom buttons
              latticeElement.innerHTML += `<div class="controls">
        <button id="zoom-in">Zoom In (+)</button>
        <button id="zoom-out">Zoom Out (-)</button>
        <button id="reset-view">Reset View</button>
        <span style="margin-left: 20px;">Zoom: <span id="zoom-level">100%</span></span>
    </div>`;
              latticeElement.innerHTML += `Lattice basis:<br/>[gx, gy] = [${alsoInCurrentTuning(state.latticeBasis[0], state.tuning, equave)}, ${alsoInCurrentTuning(state.latticeBasis[1], state.tuning, equave)}]`;
              latticeElement.appendChild(svgTag);

              // Zoom functionality
              const zoomInButton = document.getElementById("zoom-in");
              const zoomOutButton = document.getElementById("zoom-out");
              const resetViewButton = document.getElementById("reset-view");
              const zoomLevelDisplay = document.getElementById("zoom-level");
              let viewBox = {
                x: 150,
                y: 150,
                width: LATTICE_SVG_WIDTH,
                height: LATTICE_SVG_HEIGHT,
              };

              let isPanning = false;
              let isPinching = false;
              let startPoint = { x: 0, y: 0 };
              let initialPinchDistance = 0;
              let pinchCenter = { x: 0, y: 0 };
              let scale = 1.0;

              // Get distance between two touch points
              function getTouchDistance(touches) {
                const dx = touches[0].clientX - touches[1].clientX;
                const dy = touches[0].clientY - touches[1].clientY;
                return Math.sqrt(dx * dx + dy * dy);
              }

              // Get center point between two touches in SVG coordinates
              function getTouchCenter(touches) {
                const CTM = svgTag.getScreenCTM();
                const centerX = (touches[0].clientX + touches[1].clientX) / 2;
                const centerY = (touches[0].clientY + touches[1].clientY) / 2;
                return {
                  x: (centerX - CTM.e) / CTM.a,
                  y: (centerY - CTM.f) / CTM.d,
                };
              }

              // Update viewBox attribute
              function updateViewBox() {
                svgTag.setAttribute(
                  "viewBox",
                  `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`,
                );
                zoomLevelDisplay.textContent = Math.round(scale * 100) + "%";
              }

              // Convert screen coordinates to SVG coordinates
              function getPointInSVG(e) {
                const CTM = svgTag.getScreenCTM();
                // Handle both mouse and touch events
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                const clientY = e.touches ? e.touches[0].clientY : e.clientY;
                return {
                  x: (clientX - CTM.e) / CTM.a,
                  y: (clientY - CTM.f) / CTM.d,
                };
              }

              // Pointer event tracking for pinch-to-zoom
              const pointers = new Map();

              function getPointerDistance() {
                const pts = Array.from(pointers.values());
                if (pts.length < 2) return 0;
                const dx = pts[0].clientX - pts[1].clientX;
                const dy = pts[0].clientY - pts[1].clientY;
                return Math.sqrt(dx * dx + dy * dy);
              }

              function getPointerCenter() {
                const pts = Array.from(pointers.values());
                if (pts.length < 2) return { x: 0, y: 0 };
                const CTM = svgTag.getScreenCTM();
                const centerX = (pts[0].clientX + pts[1].clientX) / 2;
                const centerY = (pts[0].clientY + pts[1].clientY) / 2;
                return {
                  x: (centerX - CTM.e) / CTM.a,
                  y: (centerY - CTM.f) / CTM.d,
                };
              }

              // Pointer down
              svgTag.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                pointers.set(e.pointerId, e);

                if (pointers.size === 1) {
                  isPanning = true;
                  isPinching = false;
                  startPoint = getPointInSVG(e);
                } else if (pointers.size === 2) {
                  isPanning = false;
                  isPinching = true;
                  initialPinchDistance = getPointerDistance();
                  pinchCenter = getPointerCenter();
                }
              });

              // Pointer move
              svgTag.addEventListener("pointermove", (e) => {
                if (!pointers.has(e.pointerId)) return;

                e.preventDefault();
                pointers.set(e.pointerId, e);

                if (pointers.size === 2 && isPinching) {
                  const currentDistance = getPointerDistance();
                  if (initialPinchDistance > 0 && currentDistance > 0) {
                    const zoomFactor = initialPinchDistance / currentDistance;
                    const center = getPointerCenter();

                    // Calculate new dimensions
                    const newWidth = viewBox.width * zoomFactor;
                    const newHeight = viewBox.height * zoomFactor;

                    // Adjust position to zoom towards pinch center
                    viewBox.x += (pinchCenter.x - viewBox.x) * (1 - zoomFactor);
                    viewBox.y += (pinchCenter.y - viewBox.y) * (1 - zoomFactor);

                    viewBox.width = newWidth;
                    viewBox.height = newHeight;

                    scale = 800 / viewBox.width;

                    // Update for next frame
                    initialPinchDistance = currentDistance;
                    pinchCenter = center;

                    updateViewBox();
                  }
                } else if (pointers.size === 1 && isPanning) {
                  const CTM = svgTag.getScreenCTM();
                  const currentPoint = {
                    x: (e.clientX - CTM.e) / CTM.a,
                    y: (e.clientY - CTM.f) / CTM.d,
                  };
                  const dx = currentPoint.x - startPoint.x;
                  const dy = currentPoint.y - startPoint.y;

                  viewBox.x -= dx;
                  viewBox.y -= dy;

                  updateViewBox();
                }
              });

              // Pointer up/cancel
              function onPointerUp(e) {
                pointers.delete(e.pointerId);
                if (pointers.size < 2) {
                  isPinching = false;
                }
                if (pointers.size === 0) {
                  isPanning = false;
                }
                // If we went from 2 to 1 pointer, restart panning from current position
                if (pointers.size === 1) {
                  isPanning = true;
                  const remainingPointer = Array.from(pointers.values())[0];
                  const CTM = svgTag.getScreenCTM();
                  startPoint = {
                    x: (remainingPointer.clientX - CTM.e) / CTM.a,
                    y: (remainingPointer.clientY - CTM.f) / CTM.d,
                  };
                }
              }

              svgTag.addEventListener("pointerup", onPointerUp);
              svgTag.addEventListener("pointercancel", onPointerUp);
              svgTag.addEventListener("pointerleave", onPointerUp);

              // Wheel - zoom
              svgTag.addEventListener("wheel", (e) => {
                e.preventDefault();

                const point = getPointInSVG(e);
                const zoomFactor = e.deltaY < 0 ? 0.9 : 1.1;

                // Calculate new dimensions
                const newWidth = viewBox.width * zoomFactor;
                const newHeight = viewBox.height * zoomFactor;

                // Adjust position to zoom towards mouse cursor
                viewBox.x += (point.x - viewBox.x) * (1 - zoomFactor);
                viewBox.y += (point.y - viewBox.y) * (1 - zoomFactor);

                viewBox.width = newWidth;
                viewBox.height = newHeight;

                scale = 800 / viewBox.width;

                updateViewBox();
              });

              // Zoom by factor around center
              function zoomBy(factor) {
                const centerX = viewBox.x + viewBox.width / 2;
                const centerY = viewBox.y + viewBox.height / 2;
                viewBox.width *= factor;
                viewBox.height *= factor;
                viewBox.x = centerX - viewBox.width / 2;
                viewBox.y = centerY - viewBox.height / 2;
                scale = 800 / viewBox.width;
                updateViewBox();
              }

              // Button functions
              zoomInButton.addEventListener("click", () => zoomBy(0.8));
              zoomOutButton.addEventListener("click", () => zoomBy(1.25));
              resetViewButton.addEventListener("click", () => {
                viewBox = {
                  x: 150,
                  y: 150,
                  width: LATTICE_SVG_WIDTH,
                  height: LATTICE_SVG_HEIGHT,
                };
                scale = 1;
                updateViewBox();
              });
              // Initialize
              updateViewBox();
            } else {
              // No lattice basis available - show a message instead of throwing
              latticeElement.innerHTML = `<hr/><h2>Lattice view</h2><br/><small>No suitable lattice basis found for this scale.</small>`;
            }
          } else {
            // No word selected
            latticeElement.innerHTML = "";
          }
        }
      }
    }

    // Function for showing the SonicWeave code
    function showSonicWeaveCode(state) {
      if (state.word) {
        const arity = new Set(Array.from(state.word)).size;
        if (state.tuning) {
          const element = document.getElementById("sw-code");
          if (element) {
            element.innerHTML = `<hr/><h2>SonicWeave code</h2>
        (for <a href="https://sw3.lumipakkanen.com/" target="_blank">Scale Workshop 3</a>)<br/>`;
            element.innerHTML += `<pre class="language-ocaml"><code class="language-ocaml" id="codeblock"></code></pre>`;

            // Make a "copy to clipboard" button
            const copyButtonLabel = "Copy";

            async function copyCode(block, button) {
              let code = block.querySelector("code");
              let text = code.innerText;

              await navigator.clipboard.writeText(text);

              // visual feedback that task is completed
              button.innerText = "Copied!";

              setTimeout(() => {
                button.innerText = copyButtonLabel;
              }, 700);
            }

            // use a class selector if available
            let blocks = document.querySelectorAll("pre");

            blocks.forEach((block) => {
              // only add button if browser supports Clipboard API
              if (navigator.clipboard) {
                let button = document.createElement("button");

                button.innerText = copyButtonLabel;
                block.appendChild(button);

                button.addEventListener("click", async () => {
                  await copyCode(block, button);
                });
              }
            });

            let codeblock = document.getElementById("codeblock");
            const arr = Array.from(state.word);
            if (codeblock) {
              codeblock.innerHTML =
                arity === 3
                  ? `let L = ${state.tuning[0]}
let m = ${state.tuning[1]}
let s = ${state.tuning[2]}
${arr.join(";")};
stack()`
                  : arity === 2
                    ? `let L = ${state.tuning[0]}
let s = ${state.tuning[1]}
${arr.join(";")};
stack()`
                    : arity === 1
                      ? `let X = ${state.tuning[0]}
${arr.join(";")};
stack()`
                      : "Scales of rank > 3 are not supported";
            }
          }
        }
      }
    }

    function showScaleProfile(state, equave) {
      const el = document.getElementById("scale-profile");
      if (el) {
        el.innerHTML = "";
        const h2 = document.createElement("h2");
        h2.innerText = `Scale information for ${state.word}`;
        el.appendChild(h2);
        if (state.profile) {
          // const ploidacot = state.profile["ploidacot"];

          const [ed1, ed2, ed3] = state.profile["ed_join"];
          // Ed join (always shown)
          el.innerHTML += `Temp-agnostic ed join: ${ed1} & ${ed2} & ${ed3}<br/>`;

          const structure = state.profile["structure"];

          // MOS substitution properties (always shown)
          const a = countChar(state.word, "L");
          const b = countChar(state.word, "m");
          const c = countChar(state.word, "s");

          el.innerHTML += `<br/><b><a href="https://xenreference.com/w/MOS_substitution" target="_blank">MOS substitution</a> properties</b><br/>`;
          el.innerHTML += state.profile["subst_l_ms"]
            ? `subst ${a}L(${b}m${c}s)<br/>`
            : "";
          el.innerHTML += state.profile["subst_m_ls"]
            ? `subst ${b}m(${a}L${c}s)<br/>`
            : "";
          el.innerHTML += state.profile["subst_s_lm"]
            ? `subst ${c}s(${a}L${b}m)<br/>`
            : "";
          if (
            !state.profile["subst_l_ms"] &&
            !state.profile["subst_m_ls"] &&
            !state.profile["subst_s_lm"]
          ) {
            el.innerHTML += `None<br/>`;
          }

          // Monotone MOS properties (always shown)
          el.innerHTML += `<br/><b><a href="https://en.xen.wiki/w/Monotone-MOS_scale" target="_blank">Monotone MOS properties</a></b><br/><small>`;
          el.innerHTML += state.profile["lm"] ? `L = m<br/>` : "";
          el.innerHTML += state.profile["ms"] ? `m = s<br/>` : "";
          el.innerHTML += state.profile["s0"] ? `s = 0<br/>` : "";
          if (
            !state.profile["lm"] &&
            !state.profile["ms"] &&
            !state.profile["s0"]
          ) {
            el.innerHTML += `None<br/>`;
          }

          // Chirality (always shown)
          if (state.profile["chirality"] === "Achiral") {
            el.innerHTML += `<br/><a href="https://en.xen.wiki/w/Chirality" target="_blank">Chirality</a>: Achiral`;
          } else {
            el.innerHTML += `<br/><a href="https://en.xen.wiki/w/Chirality" target="_blank">Chirality</a>: ${state.profile["chirality"]} (reversed: ${state.profile["reversed"]})`;
          }

          // Maximum variety (always shown)
          el.innerHTML += `<br/><br/><a href="https://en.xen.wiki/w/Maximum_variety" target="_blank">Maximum variety</a>: ${state.profile["mv"]}<br/>`;

          // Guide frame info (only if structure exists)
          if (structure) {
            el.innerHTML += `<br/><b><a href="https://en.xen.wiki/w/Guide_frame
           " target="_blank">Guide frame</a></b><br/><small>`;
            let gsDisp =
              `${structure["gs"].map((g) => ` ${alsoInCurrentTuning(g, state.tuning, equave)}`)}`.slice(
                1,
              );
            el.innerHTML += `Guided <a href="https://en.xen.wiki/w/Generator_sequence" target="_blank">generator sequence</a> of ${stepVectorLength(structure["gs"][0])}-steps: GS(${gsDisp})<br/>`; // TODO prettify
            el.innerHTML += `Aggregate generator ${alsoInCurrentTuning(structure["aggregate"], state.tuning, equave)}<br/>`; // TODO prettify
            el.innerHTML += `Offsets ${structure["offset_chord"].map((g) => alsoInCurrentTuning(g, state.tuning, equave))}<br/>`; // TODO prettify
            el.innerHTML += `Multiplicity ${JSON.stringify(structure["multiplicity"])}<br/>`; // TODO prettify
          } else {
            el.innerHTML += `<b><a href="https://en.xen.wiki/w/Guide_frame" target="_blank">Guide frame</a></b><br/><small>No guide frame found.<br/><br/></small>`;
          }
        }
      }
    }

    // display both the step vector in sum form and what interval it is in the current tuning
    function alsoInCurrentTuning(v, tuning, equave) {
      if (tuning["0"].includes("\\")) {
        if (equave.num !== 2 || equave.den !== 1) {
          const str0 = tuning["0"].substring(0, tuning["0"].indexOf("<"));
          const str1 = tuning["1"].substring(0, tuning["1"].indexOf("<"));
          const str2 = tuning["2"].substring(0, tuning["2"].indexOf("<"));
          const [numLstr, denLstr] = str0.split("\\");
          const [numMstr, _1] = str1.split("\\");
          const [numSstr, _2] = str2.split("\\");
          const numL = Number(numLstr);
          const numM = Number(numMstr);
          const numS = Number(numSstr);
          const ed = Number(denLstr);
          return `${displayStepVector(v)} (${numL * (v["0"] ?? 0) + numM * (v["1"] ?? 0) + numS * (v["2"] ?? 0)}\\${ed}<${equave.num}/${equave.den}>)`;
        } else {
          const [numLstr, denLstr] = tuning["0"].split("\\");
          const ed = Number(denLstr);
          const [numMstr, _3] = tuning["1"].split("\\");
          const [numSstr, _4] = tuning["2"].split("\\");
          const numL = Number(numLstr);
          const numM = Number(numMstr);
          const numS = Number(numSstr);
          return `${displayStepVector(v)} (${numL * (v["0"] ?? 0) + numM * (v["1"] ?? 0) + numS * (v["2"] ?? 0)}\\${ed})`;
        }
      } else if (
        tuning["0"].includes("/") &&
        tuning["1"].includes("/") &&
        tuning["2"].includes("/")
      ) {
        const [numLstr, denLstr] = tuning["0"].split("/");
        const [numMstr, denMstr] = tuning["1"].split("/");
        const [numSstr, denSstr] = tuning["2"].split("/");
        const numL = BigInt(numLstr);
        const numM = BigInt(numMstr);
        const numS = BigInt(numSstr);
        const denL = BigInt(denLstr);
        const denM = BigInt(denMstr);
        const denS = BigInt(denSstr);
        const num =
          numL ** BigInt(v["0"] ?? 0) *
          numM ** BigInt(v["1"] ?? 0) *
          numS ** BigInt(v["2"] ?? 0);
        const den =
          denL ** BigInt(v["0"] ?? 0) *
          denM ** BigInt(v["1"] ?? 0) *
          denS ** BigInt(v["2"] ?? 0);
        const d = gcdBigInt(num, den);
        return `${displayStepVector(v)} (${num / d}/${den / d})`;
      } else {
        return displayStepVector(v);
      }
    }

    /**
     * Get pitch info for a scale degree given step counts
     * @param stepCounts - Object with counts of L, m, s steps: { L: n, m: n, s: n }
     * @param tuning - The current tuning object
     * @param equave - The equave { num, den, ratio }
     * @returns { pitch: string, cents: number }
     */
    function getPitchInfo(stepCounts, tuning, equave) {
      if (!tuning) return { pitch: "", cents: 0 };

      const nL = stepCounts.L || 0;
      const nM = stepCounts.m || 0;
      const nS = stepCounts.s || 0;

      // Calculate equave in cents
      const equaveCents = 1200 * Math.log2(equave.num / equave.den);

      if (tuning["0"].includes("\\")) {
        // ED tuning format like "5\12" or "5\12<3/1>"
        let str0 = tuning["0"];
        let str1 = tuning["1"];
        let str2 = tuning["2"];

        // Strip equave suffix if present
        if (str0.includes("<")) {
          str0 = str0.substring(0, str0.indexOf("<"));
          str1 = str1.substring(0, str1.indexOf("<"));
          str2 = str2.substring(0, str2.indexOf("<"));
        }

        const [numLstr, denLstr] = str0.split("\\");
        const [numMstr] = str1.split("\\");
        const [numSstr] = str2.split("\\");
        const stepsL = Number(numLstr);
        const stepsM = Number(numMstr);
        const stepsS = Number(numSstr);
        const ed = Number(denLstr);

        const totalSteps = nL * stepsL + nM * stepsM + nS * stepsS;
        const cents = (totalSteps / ed) * equaveCents;

        let pitch;
        if (equave.num !== 2 || equave.den !== 1) {
          pitch = `${totalSteps}\\${ed}<${equave.num}/${equave.den}>`;
        } else {
          pitch = `${totalSteps}\\${ed}`;
        }
        return { pitch, cents };
      } else if (
        tuning["0"].includes("/") &&
        tuning["1"].includes("/") &&
        tuning["2"].includes("/")
      ) {
        // JI tuning format like "9/8"
        const [numLstr, denLstr] = tuning["0"].split("/");
        const [numMstr, denMstr] = tuning["1"].split("/");
        const [numSstr, denSstr] = tuning["2"].split("/");
        const numL = BigInt(numLstr);
        const numM = BigInt(numMstr);
        const numS = BigInt(numSstr);
        const denL = BigInt(denLstr);
        const denM = BigInt(denMstr);
        const denS = BigInt(denSstr);

        const num =
          numL ** BigInt(nL) * numM ** BigInt(nM) * numS ** BigInt(nS);
        const den =
          denL ** BigInt(nL) * denM ** BigInt(nM) * denS ** BigInt(nS);
        const d = gcdBigInt(num, den);
        const reducedNum = num / d;
        const reducedDen = den / d;
        const ratio = Number(reducedNum) / Number(reducedDen);
        const cents = 1200 * Math.log2(ratio);
        return { pitch: `${reducedNum}/${reducedDen}`, cents };
      } else {
        // Fallback: try to compute cents from monzos or ratios
        let cents = 0;
        const tuningValues = [
          { str: tuning["0"], count: nL },
          { str: tuning["1"], count: nM },
          { str: tuning["2"], count: nS },
        ];
        for (const { str, count } of tuningValues) {
          if (str.includes("/")) {
            // It's a ratio
            const [numStr, denStr] = str.split("/");
            const ratio = Number(numStr) / Number(denStr);
            cents += count * 1200 * Math.log2(ratio);
          } else {
            // Try parsing as monzo
            const monzoCents = monzoToCents(str);
            if (monzoCents !== null) {
              cents += count * monzoCents;
            }
          }
        }
        return { pitch: `${Math.round(cents)}¢`, cents };
      }
    }

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll(`'`, "&#39;")
        .replaceAll(`"`, "&quot;");
    }

    /**
     * Clear selection from both tuning tables and select a row
     */
    function selectTuningRow(jiTable, edTable, row) {
      jiTable.querySelector(`.selected`)?.classList.remove("selected");
      edTable.querySelector(`.selected`)?.classList.remove("selected");
      row.classList.add("selected");
    }

    // Helper to update all views
    function updateViews(equave) {
      showScaleProfile(appState, equave);
      createLatticeView(appState, equave);
      showSonicWeaveCode(appState);
    }

    // UI messages for invalid input
    const ONLY_TERNARY_SCALES =
      "Only ternary (3 step sizes) scales are supported.";
    const INVALID_SCALE_WORD =
      "Scale word provided is not ternary with L, m, s. Make sure the scale word has no spaces.";
    const NO_SCALE_WORD = "No scale word provided.";
    const NO_STEP_SIGNATURE = "No step signature specified.";

    const btnSig = document.getElementById("btn-sig");
    const btnWord = document.getElementById("btn-word");

    btnSig.addEventListener("click", () => {
      const sigQuery = document.getElementById("input-step-sig").value;
      let sig = `${sigQuery}`
        .split(" ")
        .map((str) => str.trim())
        .filter((str) => str.length > 0)
        .map((str) => Number(str))
        .filter((n) => !isNaN(n) && n >= 0);

      const arity = sig.filter((m) => m > 0).length;
      const scaleSize = sig.reduce((acc, m) => (acc += m), 0);
      try {
        if (scaleSize === 0) {
          statusElement.textContent = NO_STEP_SIGNATURE;
        } else if (arity != 3) {
          statusElement.textContent = ONLY_TERNARY_SCALES;
        } else {
          if (true) {
            updateUrlForSig(sigQuery);
            statusElement.textContent = "Computing...";

            document.getElementById("tables").innerHTML = `
      <hr /><h2>Tables</h2>
      <div
        style="
          overflow-y: auto;
          overflow-x: auto;
          vertical-align: top;
          height: 500px;
          width: 100%;
        "
      >
      <div class="tables-row">
                      <div class="table-column">
                        Scales
                        <div
                          style="
                            overflow-y: auto;
                            overflow-x: auto;
                            vertical-align: top;
                            height: 420px;
                          "
                        >
                          <table class="data" id="table-scales"></table>
                        </div>
                      </div>
                      <div class="table-column">
                        ed(equave) tunings
                        <div
                          style="
                            overflow-y: auto;
                            overflow-x: auto;
                            vertical-align: top;
                            height: 420px;
                          "
                        >
                          <table class="data" id="table-ed-tunings"></table>
                        </div>
                      </div>
                      <div class="table-column">
                        JI tunings
                        <div
                          style="
                            overflow-y: auto;
                            overflow-x: auto;
                            vertical-align: top;
                            height: 420px;
                          "
                        >
                          <table class="data" id="table-ji-tunings"></table>
                        </div>
                      <button id="more-sols">Get more JI tunings</button>
                      </div>
                    </div>
                  </div>`;
            const scaleTable = document.getElementById("table-scales");
            const jiTuningTable = document.getElementById("table-ji-tunings");
            const edTuningTable = document.getElementById("table-ed-tunings");
            const equave = getEquaveRatio();
            const sigResultData = wasm.sig_result(
              sig,
              document.getElementById("monotone-lm").checked,
              document.getElementById("monotone-ms").checked,
              document.getElementById("monotone-s0").checked,
              Number(document.getElementById("ggs-len").value),
              document.querySelector('input[name="ggs-len-constraint"]:checked')
                .value,
              Number(document.getElementById("mv").value),
              document.querySelector('input[name="mv-constraint"]:checked')
                .value,
              document.querySelector('input[name="scale-type"]:checked').value,
              equave.num,
              equave.den,
              getEdBound(),
              getSLower(),
              getSUpper(),
            );
            const scales = sigResultData["profiles"].map((j) => j["word"]);
            const latticeBases = sigResultData["profiles"].map(
              (j) => j["lattice_basis"],
            );
            const profiles = sigResultData["profiles"];

            const jiTunings = sigResultData["ji_tunings"];
            const edTunings = sigResultData["ed_tunings"];
            let letters;
            if (arity === 3) {
              letters = ["L", "m", "s"];
            } else if (arity === 2) {
              letters = ["L", "s"];
            } else if (arity === 1) {
              letters = ["X"];
            } else {
              letters = [...Array(arity).keys()].map((i) => `X${i}`);
            }
            statusElement.innerHTML = `<h1>Results for ${escapeHtml([...Array(arity).keys()].map((i) => `${sig[i]}${letters[i]}`).join(""))}</h1> (click on a table row to select a scale or a tuning)`;
            makeTable(scaleTable, scales, "scale");
            // add event listener for each non-head row
            const scaleRows = scaleTable.getElementsByTagName("tr");
            if (scaleRows.length >= 3) {
              scaleRows[2].classList.add("selected"); // For some reason 2 is the first row of a nonempty table.
              appState.word = scales[0];
              appState.profile = profiles[0];
              appState.latticeBasis = appState.profile["lattice_basis"];

              for (let i = 2; i < scaleRows.length; ++i) {
                scaleRows[i].addEventListener("click", async () => {
                  // unselect selected row in either JI tuning table or ED tuning table
                  scaleTable
                    .querySelector(`.selected`)
                    .classList.remove("selected");

                  // select the row clicked on
                  scaleRows[i].classList.add("selected");
                  // get scale pattern
                  let scaleWord = scales[i - 2];
                  appState.profile = profiles[i - 2];
                  appState.latticeBasis = latticeBases[i - 2];
                  appState.word = scaleWord;
                  updateViews(equave);
                });
              }
            }
            makeTable(jiTuningTable, jiTunings);
            const jiRows = jiTuningTable.getElementsByTagName("tr");
            for (let i = 2; i < jiRows.length; ++i) {
              const thisRow = jiRows[i];
              thisRow.addEventListener("click", () => {
                if (arity === 3) {
                  selectTuningRow(jiTuningTable, edTuningTable, thisRow);
                  appState.tuning = jiTunings[i - 2];
                  updateViews(equave);
                } else {
                  statusElement.textContent = ONLY_TERNARY_SCALES;
                }
              });
            }
            if (edTunings) {
              makeTable(edTuningTable, edTunings);
              const edRows = edTuningTable.getElementsByTagName("tr");
              edRows[2].classList.add("selected");
              const edTuning = edTunings[0];
              appState.tuning = edTuning;
              updateViews(equave);
              for (let i = 2; i < edRows.length; ++i) {
                const thisRow = edRows[i];
                thisRow.addEventListener("click", () => {
                  selectTuningRow(jiTuningTable, edTuningTable, thisRow);
                  const edTuning = edTunings[i - 2];
                  appState.tuning = edTuning;
                  updateViews(equave);
                });
              }
            }

            // More JI solutions button handler
            let currentJiTunings = jiTunings;
            const moreSolsBtn = document.getElementById("more-sols");
            if (moreSolsBtn) {
              moreSolsBtn.addEventListener("click", () => {
                moreSolsBtn.textContent = "Computing...";
                moreSolsBtn.disabled = true;
                setTimeout(() => {
                  try {
                    currentJiTunings = wasm.more_ji_tunings(
                      sig,
                      equave.num,
                      equave.den,
                      getSLower(),
                      getSUpper(),
                    );
                    // Re-render the JI tuning table
                    jiTuningTable.innerHTML = "";
                    makeTable(jiTuningTable, currentJiTunings);
                    const newJiRows = jiTuningTable.getElementsByTagName("tr");
                    for (let i = 2; i < newJiRows.length; ++i) {
                      const thisRow = newJiRows[i];
                      thisRow.addEventListener("click", () => {
                        if (arity === 3) {
                          selectTuningRow(
                            jiTuningTable,
                            edTuningTable,
                            thisRow,
                          );
                          appState.tuning = currentJiTunings[i - 2];
                          updateViews(equave);
                        } else {
                          statusElement.textContent = ONLY_TERNARY_SCALES;
                        }
                      });
                    }
                    moreSolsBtn.textContent = `Get more JI tunings (${currentJiTunings.length} total)`;
                    moreSolsBtn.disabled = false;
                  } catch (err) {
                    moreSolsBtn.textContent = "Error - try again";
                    moreSolsBtn.disabled = false;
                    console.error(err);
                  }
                }, 10);
              });
            }

            updateViews(equave);
          }
        }
      } catch (err) {
        statusElement.innerText = err;
      }
    });
    btnWord.addEventListener("click", () => {
      const query = document.getElementById("input-word").value;
      const arity = new Set(Array.from(query)).size;
      const queryIsValid = arity === 3 && /^[Lms]*$/.test(query);
      if (queryIsValid) {
        updateUrlForWord(query);
        statusElement.textContent = "Computing...";
        const equave = getEquaveRatio();
        const edBound = getEdBound();
        const sLower = getSLower();
        const sUpper = getSUpper();
        const wordResultData = wasm.word_result(
          query,
          equave.num,
          equave.den,
          edBound,
          sLower,
          sUpper,
        );
        const profile = wordResultData["profile"];
        const brightestMode = wordResultData["profile"]["word"];
        const jiTunings = wordResultData["ji_tunings"];
        const edTunings = wordResultData["ed_tunings"];
        document.getElementById("tables").innerHTML = `
                  <div class="tables-row">
                    <div class="table-column">
                      ed(equave) tunings
                      <div
                        style="
                          overflow-y: auto;
                          overflow-x: auto;
                          vertical-align: top;
                          height: 420px;
                        "
                      >
                        <table class="data" id="table-ed-tunings"></table>
                      </div>
                    </div>
                    <div class="table-column">
                      JI tunings
                      <div
                        style="
                          overflow-y: auto;
                          overflow-x: auto;
                          vertical-align: top;
                          height: 420px;
                        "
                      >
                        <table class="data" id="table-ji-tunings"></table>

                      </div>
                      <button id="more-sols">Get more JI tunings (slow)</button>
                    </div>
                  </div>`;
        const jiTuningTable = document.getElementById("table-ji-tunings");
        const edTuningTable = document.getElementById("table-ed-tunings");
        appState.word = brightestMode;
        try {
          statusElement.innerHTML = `<h1>Results for ${appState.word}</h1>(click on a table row to select a tuning)`;

          makeTable(jiTuningTable, jiTunings);
          const jiRows = jiTuningTable.getElementsByTagName("tr");
          for (let i = 2; i < jiRows.length; ++i) {
            const thisRow = jiRows[i];
            thisRow.addEventListener("click", () => {
              if (arity === 3) {
                selectTuningRow(jiTuningTable, edTuningTable, thisRow);
                appState.tuning = jiTunings[i - 2];
                updateViews(equave);
              } else {
                statusElement.textContent = ONLY_TERNARY_SCALES;
              }
            });
          }
          if (edTunings) {
            makeTable(edTuningTable, edTunings);
            const edRows = edTuningTable.getElementsByTagName("tr");
            edRows[2].classList.add("selected");
            for (let i = 2; i < edRows.length; ++i) {
              const thisRow = edRows[i];
              thisRow.addEventListener("click", () => {
                selectTuningRow(jiTuningTable, edTuningTable, thisRow);
                const edTuning = edTunings[i - 2];
                appState.tuning = edTuning;
                updateViews(equave);
              });
            }
          }

          // More JI solutions button handler for word mode
          let currentJiTunings = jiTunings;
          const moreSolsBtn = document.getElementById("more-sols");
          if (moreSolsBtn) {
            // Compute step signature from word
            const stepSig = [
              countChar(query, "L"),
              countChar(query, "m"),
              countChar(query, "s"),
            ];
            moreSolsBtn.addEventListener("click", () => {
              moreSolsBtn.textContent = "Computing...";
              moreSolsBtn.disabled = true;
              setTimeout(() => {
                try {
                  const moreJiTunings = wasm.more_ji_tunings(
                    stepSig,
                    equave.num,
                    equave.den,
                    getSLower(),
                    getSUpper(),
                  );
                  // Merge with existing tunings (union by string comparison)
                  const existingSet = new Set(
                    currentJiTunings.map((t) => JSON.stringify(t)),
                  );
                  for (const tuning of moreJiTunings) {
                    const key = JSON.stringify(tuning);
                    if (!existingSet.has(key)) {
                      existingSet.add(key);
                      currentJiTunings.push(tuning);
                    }
                  }
                  // Re-render the JI tuning table
                  jiTuningTable.innerHTML = "";
                  makeTable(jiTuningTable, currentJiTunings);
                  const newJiRows = jiTuningTable.getElementsByTagName("tr");
                  for (let i = 2; i < newJiRows.length; ++i) {
                    const thisRow = newJiRows[i];
                    thisRow.addEventListener("click", () => {
                      if (arity === 3) {
                        selectTuningRow(jiTuningTable, edTuningTable, thisRow);
                        appState.tuning = currentJiTunings[i - 2];
                        updateViews(equave);
                      } else {
                        statusElement.textContent = ONLY_TERNARY_SCALES;
                      }
                    });
                  }
                  moreSolsBtn.textContent = `Get more JI tunings (${currentJiTunings.length} total)`;
                  moreSolsBtn.disabled = false;
                } catch (err) {
                  moreSolsBtn.textContent = "Error - try again";
                  moreSolsBtn.disabled = false;
                  console.error(err);
                }
              }, 10);
            });
          }

          appState.tuning = edTunings[0];
          appState.profile = profile;
          appState.latticeBasis = appState.profile["lattice_basis"];
          updateViews(equave);
        } catch (err) {
          statusElement.innerText = err;
        }
      } else if (query) {
        statusElement.textContent = INVALID_SCALE_WORD;
      } else {
        statusElement.textContent = NO_SCALE_WORD;
      }
    });

    // URL parameter handling
    function parseUrlParams() {
      const params = new URLSearchParams(window.location.search);
      return {
        word: params.get("word"),
        sig: params.get("sig"),
        equave: params.get("equave"),
        ed: params.get("ed"),
        smin: params.get("smin"),
        smax: params.get("smax"),
        type: params.get("type"),
        lm: params.get("lm"),
        ms: params.get("ms"),
        s0: params.get("s0"),
        ggs: params.get("ggs"),
        ggsmode: params.get("ggsmode"),
        mv: params.get("mv"),
        mvmode: params.get("mvmode"),
      };
    }

    function updateUrlForWord(word) {
      const params = new URLSearchParams();
      params.set("word", word);

      // Only include non-default global config values
      const equave = document.getElementById("input-equave").value.trim();
      if (equave && equave !== "2/1") params.set("equave", equave);

      const ed = document.getElementById("input-ed-bound").value;
      if (ed && ed !== "111") params.set("ed", ed);

      const smin = document.getElementById("input-s-lower").value;
      if (smin && smin !== "20") params.set("smin", smin);

      const smax = document.getElementById("input-s-upper").value;
      if (smax && smax !== "250") params.set("smax", smax);

      history.replaceState(null, "", "?" + params.toString());
    }

    function updateUrlForSig(sig) {
      const params = new URLSearchParams();
      // Convert "3 2 2" to "3+2+2"
      params.set("sig", sig.trim().replace(/\s+/g, "+"));

      // Only include non-default values
      const equave = document.getElementById("input-equave").value.trim();
      if (equave && equave !== "2/1") params.set("equave", equave);

      const ed = document.getElementById("input-ed-bound").value;
      if (ed && ed !== "111") params.set("ed", ed);

      const smin = document.getElementById("input-s-lower").value;
      if (smin && smin !== "20") params.set("smin", smin);

      const smax = document.getElementById("input-s-upper").value;
      if (smax && smax !== "250") params.set("smax", smax);

      const scaleType = document.querySelector(
        'input[name="scale-type"]:checked',
      ).value;
      if (scaleType === "all-scales") params.set("type", "all-scales");

      const lm = document.getElementById("monotone-lm").checked;
      if (lm) params.set("lm", "1");

      const ms = document.getElementById("monotone-ms").checked;
      if (ms) params.set("ms", "1");

      const s0 = document.getElementById("monotone-s0").checked;
      if (!s0) params.set("s0", "0"); // default is checked, so only set if unchecked

      const ggsLen = document.getElementById("ggs-len").value;
      if (ggsLen && ggsLen !== "0") params.set("ggs", ggsLen);

      const ggsMode = document.querySelector(
        'input[name="ggs-len-constraint"]:checked',
      ).value;
      if (ggsMode === "at-most") params.set("ggsmode", "at-most");

      const mv = document.getElementById("mv").value;
      if (mv && mv !== "0") params.set("mv", mv);

      const mvMode = document.querySelector(
        'input[name="mv-constraint"]:checked',
      ).value;
      if (mvMode === "at-most") params.set("mvmode", "at-most");

      history.replaceState(null, "", "?" + params.toString());
    }

    function populateFromUrl() {
      const params = parseUrlParams();

      // Set global config inputs if provided
      if (params.equave) {
        document.getElementById("input-equave").value = params.equave;
      }
      if (params.ed) {
        document.getElementById("input-ed-bound").value = params.ed;
      }
      if (params.smin) {
        document.getElementById("input-s-lower").value = params.smin;
      }
      if (params.smax) {
        document.getElementById("input-s-upper").value = params.smax;
      }

      // Word query takes priority
      if (params.word) {
        document.getElementById("input-word").value = params.word;
        btnWord.click();
      } else if (params.sig) {
        // Convert "3+2+2" to "3 2 2"
        document.getElementById("input-step-sig").value = params.sig.replace(
          /\+/g,
          " ",
        );

        // Set scale type radio
        if (params.type === "all-scales") {
          document.getElementById("all-scales").checked = true;
        } else {
          document.getElementById("mos-subst").checked = true;
        }

        // Set monotone checkboxes (default: lm=0, ms=0, s0=1)
        document.getElementById("monotone-lm").checked = params.lm === "1";
        document.getElementById("monotone-ms").checked = params.ms === "1";
        document.getElementById("monotone-s0").checked = params.s0 !== "0"; // default true unless explicitly "0"

        // Set GGS length
        if (params.ggs) {
          document.getElementById("ggs-len").value = params.ggs;
        }
        if (params.ggsmode === "at-most") {
          document.getElementById("ggs-at-most").checked = true;
        } else {
          document.getElementById("ggs-exactly").checked = true;
        }

        // Set MV
        if (params.mv) {
          document.getElementById("mv").value = params.mv;
        }
        if (params.mvmode === "at-most") {
          document.getElementById("mv-at-most").checked = true;
        } else {
          document.getElementById("mv-exactly").checked = true;
        }

        btnSig.click();
      }
    }

    // Auto-populate and submit from URL params on load
    populateFromUrl();
  })
  .catch((err) => {
    console.error("Failed to load WASM module:", err);
    statusElement.textContent =
      "Failed to load WASM module. Check console for details.";
  });
