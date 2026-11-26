(() => {
  "use strict";

  const H = 32;
  const W = 36;
  const CHANNELS = { t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5 };
  const API = "http://localhost:8000";
  const sample = [300, 5, 4, 2, 100800, 45];
  const ui = {
    connection: document.querySelector("#connection"),
    variable: document.querySelector("#variable"),
    day: document.querySelector("#day"),
    dayLabel: document.querySelector("#day-label"),
    play: document.querySelector("#play"),
    refresh: document.querySelector("#refresh"),
    shockSize: document.querySelector("#shock-size"),
    shockUnit: document.querySelector("#shock-unit"),
    shockStatus: document.querySelector("#shock-status"),
    error: document.querySelector("#error"),
    base: document.querySelector("#map-base"),
    shock: document.querySelector("#map-shock"),
    delta: document.querySelector("#map-delta"),
  };
  let baseline = null;
  let shocked = null;
  let perturbation = zeroShock();
  let requestId = 0;
  let playTimer = null;
  let repaintTimer = null;
  let activePreset = "none";

  function zeroShock() {
    return Array.from({ length: 6 }, () => Array.from({ length: H }, () => Array(W).fill(0)));
  }

  function makeInitialStates() {
    return Array.from({ length: 3 }, () =>
      Array.from({ length: 6 }, (_, channel) =>
        Array.from({ length: H }, () => Array(W).fill(sample[channel]))
      )
    );
  }

  function setConnection(ok, message) {
    ui.connection.textContent = message;
    ui.connection.classList.toggle("online", ok);
    ui.connection.classList.toggle("offline", !ok);
  }

  function showError(message) {
    ui.error.hidden = !message;
    ui.error.textContent = message || "";
  }

  function configureUnit() {
    const units = { tp: "mm/day", t2m: "K", wind: "m/s", msl: "Pa", tcwv: "kg/m²" };
    ui.shockUnit.textContent = units[ui.variable.value];
  }

  async function postSimulation(shock) {
    const response = await fetch(`${API}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        init_states: makeInitialStates(),
        doy_sincos: [Math.sin(2 * Math.PI * 180 / 365.25), Math.cos(2 * Math.PI * 180 / 365.25)],
        steps: 10,
        perturbation: shock,
      }),
    });
    if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);
    const data = await response.json();
    if (!data.rollout || data.rollout.length !== 10) throw new Error("API response did not contain a ten-day rollout.");
    return data.rollout;
  }

  async function runForecast() {
    const run = ++requestId;
    ui.refresh.disabled = true;
    ui.refresh.textContent = "Running…";
    showError("");
    try {
      const [base, branch] = await Promise.all([postSimulation(null), postSimulation(perturbation)]);
      if (run !== requestId) return;
      baseline = base;
      shocked = branch;
      setConnection(true, "Local model connected");
      drawAll();
      const response = await fetch(`${API}/health`, { method: "POST" });
      if (response.ok) {
        const health = await response.json();
        setConnection(true, `Ready · ${health.model_name || "local model"}`);
      }
    } catch (err) {
      if (run === requestId) {
        setConnection(false, "Local model unavailable");
        showError(`Could not reach ${API}. Start the API with “python serve.py --port 8000”, then run the forecast again. ${err.message}`);
      }
    } finally {
      if (run === requestId) {
        ui.refresh.disabled = false;
        ui.refresh.textContent = "Run forecast";
      }
    }
  }

  function selectedField(rollout, day) {
    if (!rollout) return null;
    const frame = rollout[day];
    const variable = ui.variable.value;
    if (variable === "wind") {
      return Array.from({ length: H }, (_, i) =>
        Array.from({ length: W }, (_, j) => Math.hypot(frame[CHANNELS.u10][i][j], frame[CHANNELS.v10][i][j]))
      );
    }
    return frame[CHANNELS[variable]];
  }

  function colorAt(value, low, high) {
    const stops = [
      [37, 59, 144], [25, 136, 188], [72, 187, 138], [244, 212, 77], [216, 71, 63],
    ];
    const t = Math.max(0, Math.min(0.99999, (value - low) / Math.max(1e-9, high - low)));
    const segment = t * (stops.length - 1);
    const index = Math.floor(segment);
    const mix = segment - index;
    const a = stops[index];
    const b = stops[index + 1];
    return `rgb(${a.map((n, i) => Math.round(n + (b[i] - n) * mix)).join(",")})`;
  }

  function divergingColor(value, span) {
    const stops = [[84, 53, 139], [156, 134, 197], [244, 245, 243], [230, 155, 112], [169, 55, 50]];
    const t = Math.max(0, Math.min(0.99999, 0.5 + value / (2 * span)));
    const segment = t * (stops.length - 1);
    const index = Math.floor(segment);
    const mix = segment - index;
    const a = stops[index];
    const b = stops[index + 1];
    return `rgb(${a.map((n, i) => Math.round(n + (b[i] - n) * mix)).join(",")})`;
  }

  function mapRange(...fields) {
    const values = fields.flatMap(field => field ? field.flat().filter(Number.isFinite) : []);
    if (!values.length) return [0, 1];
    let low = Math.min(...values);
    let high = Math.max(...values);
    if (low === high) { low -= 0.5; high += 0.5; }
    return [low, high];
  }

  function strokeIndia(ctx, width, height) {
    // Simplified outline for geographic orientation; it is not a boundary dataset.
    const outline = [
      [68.1, 23.8], [69.2, 22.2], [70.1, 20.8], [72.1, 19.0], [72.8, 16.4],
      [73.5, 13.6], [75.0, 10.7], [76.8, 8.1], [78.0, 9.0], [79.7, 10.7],
      [81.0, 13.0], [82.0, 15.1], [84.1, 16.2], [86.8, 18.3], [88.0, 21.0],
      [89.6, 22.1], [92.0, 23.8], [94.2, 26.1], [92.0, 27.3], [89.4, 28.1],
      [86.8, 28.3], [84.4, 29.6], [82.5, 30.6], [80.7, 31.8], [78.9, 34.1],
      [77.3, 36.9], [75.4, 34.7], [73.9, 32.5], [72.2, 30.8], [70.9, 28.6],
      [69.3, 26.3], [68.1, 23.8],
    ];
    ctx.save();
    ctx.beginPath();
    outline.forEach(([lon, lat], index) => {
      const x = (lon - 65.5) / 35 * width;
      const y = (37.5 - lat) / 31 * height;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = "rgba(18, 40, 59, .78)";
    ctx.lineWidth = Math.max(1.1, width / 550);
    ctx.stroke();
    ctx.restore();
  }

  function drawField(canvas, field, range, diverging = false) {
    const ctx = canvas.getContext("2d");
    const cw = canvas.width / W;
    const ch = canvas.height / H;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!field) {
      ctx.fillStyle = "#708191";
      ctx.font = "16px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run forecast to load maps", canvas.width / 2, canvas.height / 2);
      return;
    }
    for (let y = 0; y < H; y++) {
      const sourceRow = H - 1 - y;
      for (let x = 0; x < W; x++) {
        const value = field[sourceRow][x];
        ctx.fillStyle = diverging ? divergingColor(value, range) : colorAt(value, range[0], range[1]);
        ctx.fillRect(x * cw, y * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    ctx.strokeStyle = "rgba(255,255,255,.14)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x++) { ctx.beginPath(); ctx.moveTo(x * cw, 0); ctx.lineTo(x * cw, canvas.height); ctx.stroke(); }
    for (let y = 0; y <= H; y++) { ctx.beginPath(); ctx.moveTo(0, y * ch); ctx.lineTo(canvas.width, y * ch); ctx.stroke(); }
    strokeIndia(ctx, canvas.width, canvas.height);
  }

  function formatValue(value) {
    if (!Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    return abs >= 1000 ? value.toExponential(1) : value.toFixed(abs < 10 ? 1 : 0);
  }

  function activeDay() { return Number(ui.day.value) - 1; }

  function drawAll() {
    ui.dayLabel.textContent = `Day ${ui.day.value}`;
    const day = activeDay();
    const baseField = selectedField(baseline, day);
    const shockField = selectedField(shocked, day);
    if (!baseField || !shockField) return;
    const deltaField = baseField.map((row, i) => row.map((value, j) => shockField[i][j] - value));
    const [low, high] = mapRange(baseField, shockField);
    const deltaSpan = Math.max(0.1, ...deltaField.flat().map(Math.abs));
    drawField(ui.base, baseField, [low, high]);
    drawField(ui.shock, shockField, [low, high]);
    drawField(ui.delta, deltaField, deltaSpan, true);
    document.querySelector("#base-min").textContent = formatValue(low);
    document.querySelector("#base-max").textContent = formatValue(high);
    document.querySelector("#shock-min").textContent = formatValue(low);
    document.querySelector("#shock-max").textContent = formatValue(high);
    document.querySelector("#delta-min").textContent = `−${formatValue(deltaSpan)}`;
    document.querySelector("#delta-max").textContent = `+${formatValue(deltaSpan)}`;
    updateImpacts(day);
  }

  function updateImpacts(day) {
    if (!shocked) return;
    const frame = shocked[day];
    const count = (test) => {
      let total = 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) if (test(i, j)) total++;
      return total;
    };
    const heat = count((i, j) => frame[CHANNELS.t2m][i][j] > 313.15);
    const heavy = count((i, j) => frame[CHANNELS.tp][i][j] > 64.5 && frame[CHANNELS.tp][i][j] <= 115.5);
    const cloud = count((i, j) => frame[CHANNELS.tp][i][j] > 115.5);
    let drought = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        let rainSum = 0;
        let tempSum = 0;
        for (let step = 0; step < 10; step++) {
          rainSum += shocked[step][CHANNELS.tp][i][j];
          tempSum += shocked[step][CHANNELS.t2m][i][j];
        }
        if (rainSum < 25 && tempSum / 10 > 308.15) drought++;
      }
    }
    document.querySelector("#heat-count").textContent = heat;
    document.querySelector("#rain-count").textContent = heavy;
    document.querySelector("#cloud-count").textContent = cloud;
    document.querySelector("#drought-count").textContent = drought;
  }

  function setShockStatus() {
    const labels = {
      none: "No shock applied. Map clicks paint a local perturbation.",
      warm: "Warm Bay of Bengal preset · +2 K over 8–20°N, 82–94°E.",
      arabian: "Weak Arabian Sea wind preset · −3 m/s over 8–20°N, 65–75°E.",
      "el-nino": "El Niño heat dome preset · +2 K over central India.",
      custom: "Custom map shock active. Click another cell to add to it, or clear the shock.",
    };
    ui.shockStatus.textContent = labels[activePreset] || labels.custom;
  }

  function paintRegion({ variable, value, latMin, latMax, lonMin, lonMax }) {
    const channels = variable === "wind" ? [CHANNELS.u10, CHANNELS.v10] : [CHANNELS[variable]];
    for (let i = 0; i < H; i++) {
      const lat = 6.5 + i;
      if (lat < latMin || lat > latMax) continue;
      for (let j = 0; j < W; j++) {
        const lon = 65.5 + j;
        if (lon < lonMin || lon > lonMax) continue;
        for (const channel of channels) perturbation[channel][i][j] += value;
      }
    }
  }

  function applyPreset(name) {
    perturbation = zeroShock();
    activePreset = name;
    if (name === "warm") paintRegion({ variable: "t2m", value: 2, latMin: 8, latMax: 20, lonMin: 82, lonMax: 94 });
    if (name === "arabian") paintRegion({ variable: "wind", value: -3, latMin: 8, latMax: 20, lonMin: 65, lonMax: 75 });
    if (name === "el-nino") paintRegion({ variable: "t2m", value: 2, latMin: 16, latMax: 27, lonMin: 72, lonMax: 86 });
    setShockStatus();
    runForecast();
  }

  function paintCell(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * W);
    const displayRow = Math.floor((event.clientY - rect.top) / rect.height * H);
    const i = H - 1 - displayRow;
    if (i < 0 || i >= H || x < 0 || x >= W) return;
    const amount = Number(ui.shockSize.value);
    if (!Number.isFinite(amount) || amount === 0) return;
    const variable = ui.variable.value;
    const channels = variable === "wind" ? [CHANNELS.u10, CHANNELS.v10] : [CHANNELS[variable]];
    for (const channel of channels) perturbation[channel][i][x] += amount;
    activePreset = "custom";
    setShockStatus();
    ui.shockStatus.textContent += ` Painted cell lat ${6.5 + i}°N, lon ${65.5 + x}°E.`;
    window.clearTimeout(repaintTimer);
    repaintTimer = window.setTimeout(runForecast, 350);
  }

  function clearShock() {
    perturbation = zeroShock();
    activePreset = "none";
    setShockStatus();
    runForecast();
  }

  document.querySelectorAll(".preset").forEach(button => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  document.querySelector("#clear-shock").addEventListener("click", clearShock);
  ui.variable.addEventListener("change", () => { configureUnit(); drawAll(); });
  ui.day.addEventListener("input", drawAll);
  ui.refresh.addEventListener("click", runForecast);
  [ui.base, ui.shock, ui.delta].forEach(canvas => canvas.addEventListener("click", event => paintCell(canvas, event)));
  ui.play.addEventListener("click", () => {
    if (playTimer) {
      window.clearInterval(playTimer);
      playTimer = null;
      ui.play.textContent = "Play";
      return;
    }
    ui.play.textContent = "Pause";
    playTimer = window.setInterval(() => {
      ui.day.value = Number(ui.day.value) >= 10 ? 1 : Number(ui.day.value) + 1;
      drawAll();
    }, 850);
  });

  configureUnit();
  runForecast();
})();
