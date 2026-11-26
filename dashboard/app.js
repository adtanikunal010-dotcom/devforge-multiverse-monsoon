(() => {
  "use strict";

  const H = 32;
  const W = 36;
  const CHANNELS = { t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5 };
  const API = "http://localhost:8000";
  const SAMPLE_STATE = [300, 5, 4, 2, 100800, 45];
  const MAP_WIDTH = 10.4;
  const MAP_DEPTH = 8.9;
  const INDIA_OUTLINE = [
    [68.1, 23.8], [69.2, 22.2], [70.1, 20.8], [72.1, 19.0], [72.8, 16.4],
    [73.5, 13.6], [75.0, 10.7], [76.8, 8.1], [78.0, 9.0], [79.7, 10.7],
    [81.0, 13.0], [82.0, 15.1], [84.1, 16.2], [86.8, 18.3], [88.0, 21.0],
    [89.6, 22.1], [92.0, 23.8], [94.2, 26.1], [92.0, 27.3], [89.4, 28.1],
    [86.8, 28.3], [84.4, 29.6], [82.5, 30.6], [80.7, 31.8], [78.9, 34.1],
    [77.3, 36.9], [75.4, 34.7], [73.9, 32.5], [72.2, 30.8], [70.9, 28.6],
    [69.3, 26.3], [68.1, 23.8],
  ];
  const VARIABLES = {
    tp: { label: "PRECIPITATION", unit: "mm/day", shock: "mm/day" },
    t2m: { label: "AIR TEMPERATURE", unit: "K", shock: "K" },
    wind: { label: "WIND SPEED", unit: "m/s", shock: "m/s" },
    msl: { label: "SEA-LEVEL PRESSURE", unit: "Pa", shock: "Pa" },
    tcwv: { label: "COLUMN WATER VAPOUR", unit: "kg/m²", shock: "kg/m²" },
  };
  const REGIONS = {
    all: { label: "All India", latMin: 6.5, latMax: 37.5, lonMin: 65.5, lonMax: 100.5, center: [22, 81] },
    west: { label: "Western coast", latMin: 8, latMax: 23, lonMin: 68, lonMax: 78, center: [16, 73.5] },
    gangetic: { label: "Indo-Gangetic plain", latMin: 23, latMax: 31, lonMin: 72, lonMax: 88, center: [27.5, 81] },
    bay: { label: "Bay of Bengal", latMin: 8, latMax: 20, lonMin: 82, lonMax: 94, center: [14, 88] },
    deccan: { label: "Deccan plateau", latMin: 10, latMax: 22, lonMin: 74, lonMax: 85, center: [16, 79.5] },
  };
  const INDIA_REGIONS = ["west", "gangetic", "bay", "deccan"];
  const STATIC_MAPS = window.MONSOON_STATIC_MAPS || null;
  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  const ui = {
    connection: document.querySelector("#connection"),
    connectionText: document.querySelector("#connection span:last-child"),
    variable: document.querySelector("#variable"),
    region: document.querySelector("#region"),
    day: document.querySelector("#day"),
    dayLabel: document.querySelector("#day-label"),
    play: document.querySelector("#play"),
    refresh: document.querySelector("#refresh"),
    shockSize: document.querySelector("#shock-size"),
    shockUnit: document.querySelector("#shock-unit"),
    shockStatus: document.querySelector("#shock-status"),
    error: document.querySelector("#error"),
    mapInstructions: document.querySelector("#map-instructions"),
    mapVariableLabel: document.querySelector("#map-variable-label"),
    mapUnitsLabel: document.querySelector("#map-units-label"),
    sampleBadge: document.querySelector("#sample-badge"),
    base: document.querySelector("#map-base"),
    shock: document.querySelector("#map-shock"),
    delta: document.querySelector("#map-delta"),
    summaryRain: document.querySelector("#summary-rain"),
    summaryHeat: document.querySelector("#summary-heat"),
    summaryDelta: document.querySelector("#summary-delta"),
    summaryRegion: document.querySelector("#summary-region"),
    summaryRegionCells: document.querySelector("#summary-region-cells"),
    regionalSummary: document.querySelector("#regional-summary"),
    regionalStatus: document.querySelector("#regional-status"),
    activeRegionName: document.querySelector("#active-region-name"),
  };
  const mapViews = {};
  let baseline = null;
  let shocked = null;
  let perturbation = zeroShock();
  let requestId = 0;
  let playTimer = null;
  let repaintTimer = null;
  let activePreset = "none";
  let activeRegion = "all";

  function zeroShock() {
    return Array.from({ length: 6 }, () => Array.from({ length: H }, () => Array(W).fill(0)));
  }

  function makeInitialStates() {
    return Array.from({ length: 3 }, () =>
      Array.from({ length: 6 }, (_, channel) =>
        Array.from({ length: H }, () => Array(W).fill(SAMPLE_STATE[channel]))
      )
    );
  }

  function setConnection(ok, message) {
    ui.connection.classList.toggle("online", ok);
    ui.connection.classList.toggle("offline", !ok);
    ui.connectionText.textContent = message;
  }

  function showError(message) {
    ui.error.hidden = !message;
    ui.error.textContent = message || "";
  }

  function configureUnit() {
    const config = VARIABLES[ui.variable.value];
    ui.shockUnit.textContent = config.shock;
    ui.mapVariableLabel.textContent = config.label;
    ui.mapUnitsLabel.textContent = config.unit;
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
    ui.refresh.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> SIMULATING';
    showError("");
    try {
      const [base, branch] = await Promise.all([postSimulation(null), postSimulation(perturbation)]);
      if (run !== requestId) return;
      baseline = base;
      shocked = branch;
      setConnection(true, "LOCAL MODEL READY");
      ui.sampleBadge.classList.add("active");
      drawAll();
    } catch (error) {
      if (run === requestId) {
        setConnection(false, "MODEL OFFLINE");
        showError(`Could not reach ${API}. Start the API with “python serve.py --port 8000”, then run the forecast again. ${error.message}`);
      }
    } finally {
      if (run === requestId) {
        ui.refresh.disabled = false;
        ui.refresh.innerHTML = "<span>↻</span> RUN FORECAST";
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
    const stops = [[28, 72, 142], [35, 147, 190], [47, 202, 191], [223, 190, 99], [234, 105, 92]];
    const t = Math.max(0, Math.min(0.99999, (value - low) / Math.max(1e-9, high - low)));
    const segment = t * (stops.length - 1);
    const index = Math.floor(segment);
    const mix = segment - index;
    const a = stops[index];
    const b = stops[index + 1];
    return `rgb(${a.map((n, i) => Math.round(n + (b[i] - n) * mix)).join(",")})`;
  }

  function divergingColor(value, span) {
    const stops = [[118, 86, 206], [169, 155, 232], [204, 224, 230], [232, 154, 115], [211, 79, 82]];
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

  function pointInIndia(lon, lat) {
    let inside = false;
    for (let i = 0, j = INDIA_OUTLINE.length - 1; i < INDIA_OUTLINE.length; j = i++) {
      const [xi, yi] = INDIA_OUTLINE[i];
      const [xj, yj] = INDIA_OUTLINE[j];
      const crosses = (yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
      if (crosses) inside = !inside;
    }
    return inside;
  }

  function terrainRelief(i, j) {
    const index = i * W + j;
    const elevation = STATIC_MAPS?.elevationMeters?.[index];
    if (Number.isFinite(elevation)) return 0.06 + Math.max(0, elevation) / 5284 * 0.84;
    const lon = 65.5 + j;
    const lat = 6.5 + i;
    const gaussian = (x, center, spread) => Math.exp(-((x - center) ** 2) / (2 * spread * spread));
    const himalaya = gaussian(lat, 31.8, 2.7) * gaussian(lon, 83.5, 8.5) * 0.9;
    const westernGhats = gaussian(lon, 74.3, 1.35) * gaussian(lat, 16.5, 7.5) * 0.3;
    const deccan = gaussian(lat, 18.5, 5.5) * gaussian(lon, 79.2, 5.4) * 0.15;
    return 0.08 + himalaya + westernGhats + deccan;
  }

  function regionMatches(regionKey, i, j) {
    if (regionKey === "all") return true;
    const region = REGIONS[regionKey];
    const lat = 6.5 + i;
    const lon = 65.5 + j;
    return lat >= region.latMin && lat <= region.latMax && lon >= region.lonMin && lon <= region.lonMax;
  }

  function worldPoint(lon, lat) {
    return {
      x: ((lon - 65.5) / (W - 1) - 0.5) * MAP_WIDTH,
      z: ((37.5 - lat) / (H - 1) - 0.5) * MAP_DEPTH,
    };
  }

  function strokeIndia(ctx, width, height) {
    ctx.save();
    ctx.beginPath();
    INDIA_OUTLINE.forEach(([lon, lat], index) => {
      const x = (lon - 65.5) / 35 * width;
      const y = (37.5 - lat) / 31 * height;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = "rgba(84, 213, 237, .72)";
    ctx.lineWidth = Math.max(1.2, width / 500);
    ctx.stroke();
    ctx.restore();
  }

  function drawField(canvas, field, range, diverging = false) {
    const context = canvas.getContext("2d");
    if (!context) return;
    const cellWidth = canvas.width / W;
    const cellHeight = canvas.height / H;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#0a1b29";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (!field) {
      context.fillStyle = "#84a0b1";
      context.font = "13px -apple-system, sans-serif";
      context.textAlign = "center";
      context.fillText("Run forecast to load maps", canvas.width / 2, canvas.height / 2);
      return;
    }
    for (let y = 0; y < H; y++) {
      const row = H - 1 - y;
      for (let x = 0; x < W; x++) {
        const value = field[row][x];
        context.fillStyle = diverging ? divergingColor(value, range) : colorAt(value, range[0], range[1]);
        context.fillRect(x * cellWidth, y * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
      }
    }
    context.strokeStyle = "rgba(179, 223, 234, .11)";
    context.lineWidth = 1;
    for (let x = 0; x <= W; x++) { context.beginPath(); context.moveTo(x * cellWidth, 0); context.lineTo(x * cellWidth, canvas.height); context.stroke(); }
    for (let y = 0; y <= H; y++) { context.beginPath(); context.moveTo(0, y * cellHeight); context.lineTo(canvas.width, y * cellHeight); context.stroke(); }
    strokeIndia(context, canvas.width, canvas.height);
  }

  function createAtmosphere(T, scene, seed) {
    let state = seed >>> 0;
    const random = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; };
    const count = 54;
    const positions = new Float32Array(count * 6);
    const speeds = new Float32Array(count);
    for (let n = 0; n < count; n++) {
      let lon = 79;
      let lat = 21;
      for (let attempt = 0; attempt < 40; attempt++) {
        lon = 67 + random() * 29;
        lat = 7 + random() * 30;
        if (pointInIndia(lon, lat)) break;
      }
      const { x, z } = worldPoint(lon, lat);
      const y = 1.3 + random() * 2.5;
      const trail = 0.12 + random() * 0.2;
      positions.set([x, y, z, x + 0.015, y - trail, z], n * 6);
      speeds[n] = 0.24 + random() * 0.4;
    }
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.BufferAttribute(positions, 3));
    const rain = new T.LineSegments(geometry, new T.LineBasicMaterial({ color: 0x75def5, transparent: true, opacity: 0.2, depthWrite: false }));
    rain.renderOrder = 4;
    scene.add(rain);
    return { geometry, positions, speeds, count };
  }

  function ThreeMapView(canvas, viewIndex) {
    const T = window.THREE;
    this.canvas = canvas;
    this.viewIndex = viewIndex;
    this.scene = new T.Scene();
    this.scene.background = new T.Color("#091724");
    this.camera = new T.PerspectiveCamera(35, 1, 0.1, 80);
    this.target = new T.Vector3(0, 0.42, 0);
    this.camera.position.set(0, 8.4, 12.4);
    this.camera.lookAt(this.target);
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.setClearColor(0x091724, 0);
    this.scene.add(new T.HemisphereLight(0xc6efff, 0x172636, 1.8));
    const keyLight = new T.DirectionalLight(0xe4f9ff, 2.0);
    keyLight.position.set(-4, 10, 7);
    this.scene.add(keyLight);
    const rimLight = new T.PointLight(0x298ab8, 12, 20, 2);
    rimLight.position.set(4, 5, -4);
    this.scene.add(rimLight);

    const floor = new T.Mesh(
      new T.PlaneGeometry(MAP_WIDTH + 0.5, MAP_DEPTH + 0.5),
      new T.MeshStandardMaterial({ color: 0x10283a, roughness: 1, metalness: 0 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.08;
    this.scene.add(floor);
    const floorGrid = new T.GridHelper(MAP_WIDTH, 26, 0x31566b, 0x244354);
    floorGrid.scale.z = MAP_DEPTH / MAP_WIDTH;
    floorGrid.position.y = -0.065;
    floorGrid.material.transparent = true;
    floorGrid.material.opacity = 0.22;
    this.scene.add(floorGrid);

    const cellWidth = MAP_WIDTH / W;
    const cellDepth = MAP_DEPTH / H;
    this.cellWidth = cellWidth;
    this.cellDepth = cellDepth;
    const cellGeometry = new T.BoxGeometry(cellWidth * 0.93, 1, cellDepth * 0.93);
    const cellMaterial = new T.MeshStandardMaterial({ color: 0xffffff, roughness: 0.72, metalness: 0.02 });
    this.surface = new T.InstancedMesh(cellGeometry, cellMaterial, H * W);
    this.surface.instanceMatrix.setUsage(T.DynamicDrawUsage);
    this.visibleCells = new Uint8Array(H * W);
    this.currentHeights = new Float32Array(H * W);
    this.currentColors = Array.from({ length: H * W }, () => new T.Color(0x15516b));
    const hiddenCell = new T.Object3D();
    const emptyColor = new T.Color(0x10283a);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const index = i * W + j;
        const landFraction = STATIC_MAPS?.landMaskMilli?.[index];
        const isLand = !Number.isFinite(landFraction) || landFraction >= 180;
        const inside = isLand && pointInIndia(65.5 + j, 6.5 + i);
        this.visibleCells[index] = inside ? 1 : 0;
        hiddenCell.position.set(0, -1, 0);
        hiddenCell.scale.set(0, 0, 0);
        hiddenCell.updateMatrix();
        this.surface.setMatrixAt(index, hiddenCell.matrix);
        this.surface.setColorAt(index, emptyColor);
      }
    }
    this.surface.instanceMatrix.needsUpdate = true;
    this.surface.instanceColor.needsUpdate = true;
    this.surface.boundingSphere = new T.Sphere(new T.Vector3(0, 0.8, 0), 13);
    this.scene.add(this.surface);

    const outlinePoints = INDIA_OUTLINE.map(([lon, lat]) => {
      const { x, z } = worldPoint(lon, lat);
      return new T.Vector3(x, 1.88, z);
    });
    this.outline = new T.LineLoop(
      new T.BufferGeometry().setFromPoints(outlinePoints),
      new T.LineBasicMaterial({ color: 0x9feaf4, transparent: true, opacity: 0.82, depthTest: false })
    );
    this.outline.renderOrder = 5;
    this.scene.add(this.outline);

    const hoverGeometry = new T.EdgesGeometry(new T.BoxGeometry(cellWidth * 1.03, 1, cellDepth * 1.03));
    this.hoverOutline = new T.LineSegments(hoverGeometry, new T.LineBasicMaterial({ color: 0xc8f7ff, transparent: true, opacity: 0.94, depthTest: false }));
    this.hoverOutline.visible = false;
    this.hoverOutline.renderOrder = 6;
    this.scene.add(this.hoverOutline);
    this.regionOutline = new T.LineLoop(new T.BufferGeometry(), new T.LineBasicMaterial({ color: 0x54d5ed, transparent: true, opacity: 0.9, depthTest: false }));
    this.regionOutline.visible = false;
    this.regionOutline.renderOrder = 6;
    this.scene.add(this.regionOutline);

    this.raycaster = new T.Raycaster();
    this.pointer = new T.Vector2();
    this.drag = null;
    this.onPaint = null;
    this.onHover = null;
    this.currentField = null;
    this.selectedRegion = "all";
    this.transitionFrame = 0;
    this.cameraFrame = 0;
    this.particles = createAtmosphere(T, this.scene, 6143 + viewIndex * 941);
    this.resize = () => {
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.render();
    };
    this.render = () => this.renderer.render(this.scene, this.camera);
    this.canvas.addEventListener("pointerdown", event => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", event => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", event => this.pointerUp(event));
    this.canvas.addEventListener("pointercancel", () => this.cancelPointer());
    this.canvas.addEventListener("pointerleave", () => this.clearHover());
    this.canvas.addEventListener("wheel", event => this.zoom(event), { passive: false });
    if ("ResizeObserver" in window) new ResizeObserver(this.resize).observe(canvas);
    else window.addEventListener("resize", this.resize);
    this.resize();
  }

  ThreeMapView.prototype.toCell = function(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObject(this.surface, false)[0];
    if (!hit || !Number.isInteger(hit.instanceId) || !this.visibleCells[hit.instanceId]) return null;
    return { index: hit.instanceId, i: Math.floor(hit.instanceId / W), j: hit.instanceId % W };
  };

  ThreeMapView.prototype.pointerDown = function(event) {
    if (event.button !== 0) return;
    this.drag = { lastX: event.clientX, lastY: event.clientY, distance: 0 };
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.style.cursor = "grabbing";
  };

  ThreeMapView.prototype.pointerMove = function(event) {
    if (this.drag) {
      const dx = event.clientX - this.drag.lastX;
      const dy = event.clientY - this.drag.lastY;
      this.drag.lastX = event.clientX;
      this.drag.lastY = event.clientY;
      this.drag.distance += Math.abs(dx) + Math.abs(dy);
      if (this.drag.distance > 3) {
        const offset = this.camera.position.clone().sub(this.target);
        const spherical = new window.THREE.Spherical().setFromVector3(offset);
        spherical.theta -= dx * 0.008;
        spherical.phi = Math.max(0.35, Math.min(1.42, spherical.phi + dy * 0.008));
        this.camera.position.setFromSpherical(spherical).add(this.target);
        this.camera.lookAt(this.target);
        this.render();
      }
      return;
    }
    this.updateHover(event);
  };

  ThreeMapView.prototype.pointerUp = function(event) {
    const drag = this.drag;
    this.drag = null;
    this.canvas.style.cursor = "grab";
    if (!drag || drag.distance >= 5) return;
    const cell = this.toCell(event);
    if (cell && this.onPaint) this.onPaint(cell.i, cell.j);
  };

  ThreeMapView.prototype.cancelPointer = function() {
    this.drag = null;
    this.canvas.style.cursor = "grab";
  };

  ThreeMapView.prototype.updateHover = function(event) {
    const cell = this.toCell(event);
    if (!cell) { this.clearHover(); return; }
    const height = this.currentHeights[cell.index] || 0.2;
    const lon = 65.5 + cell.j;
    const lat = 6.5 + cell.i;
    this.hoverOutline.position.set((cell.j - (W - 1) / 2) * this.cellWidth, height / 2, ((H - 1) / 2 - cell.i) * this.cellDepth);
    this.hoverOutline.scale.set(1, Math.max(height, 0.12), 1);
    this.hoverOutline.visible = true;
    this.canvas.style.cursor = "crosshair";
    if (this.onHover) this.onHover({ lat, lon, value: this.currentField?.[cell.i]?.[cell.j] });
    this.render();
  };

  ThreeMapView.prototype.clearHover = function() {
    if (!this.hoverOutline?.visible) return;
    this.hoverOutline.visible = false;
    this.canvas.style.cursor = "grab";
    if (this.onHover) this.onHover(null);
    this.render();
  };

  ThreeMapView.prototype.zoom = function(event) {
    event.preventDefault();
    const offset = this.camera.position.clone().sub(this.target);
    const spherical = new window.THREE.Spherical().setFromVector3(offset);
    spherical.radius = Math.max(7.6, Math.min(18.5, spherical.radius * (event.deltaY > 0 ? 1.08 : 0.92)));
    this.camera.position.setFromSpherical(spherical).add(this.target);
    this.camera.lookAt(this.target);
    this.render();
  };

  ThreeMapView.prototype.update = function(field, range, diverging = false) {
    const T = window.THREE;
    const targetHeights = new Float32Array(H * W);
    const targetColors = new Array(H * W);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const index = i * W + j;
        if (!this.visibleCells[index]) { targetColors[index] = new T.Color(0x10283a); continue; }
        const value = field[i][j];
        const normalized = diverging
          ? Math.max(0, Math.min(1, 0.5 + value / (2 * range)))
          : Math.max(0, Math.min(1, (value - range[0]) / Math.max(1e-9, range[1] - range[0])));
        const terrain = terrainRelief(i, j);
        targetHeights[index] = diverging ? 0.18 + terrain * 0.25 + normalized * 0.62 : 0.14 + terrain * 0.42 + normalized * 0.86;
        targetColors[index] = new T.Color(diverging ? divergingColor(value, range) : colorAt(value, range[0], range[1]));
      }
    }
    this.currentField = field;
    if (this.transitionFrame) cancelAnimationFrame(this.transitionFrame);
    const fromHeights = this.currentHeights.slice();
    const fromColors = this.currentColors.map(color => color.clone());
    const duration = prefersReducedMotion ? 0 : 300;
    const start = performance.now();
    const apply = progress => {
      const eased = 1 - (1 - progress) ** 3;
      const object = new T.Object3D();
      const color = new T.Color();
      for (let index = 0; index < H * W; index++) {
        if (!this.visibleCells[index]) {
          object.position.set(0, -1, 0);
          object.scale.set(0, 0, 0);
          object.updateMatrix();
          this.surface.setMatrixAt(index, object.matrix);
          this.surface.setColorAt(index, targetColors[index]);
          continue;
        }
        const height = fromHeights[index] + (targetHeights[index] - fromHeights[index]) * eased;
        this.currentHeights[index] = height;
        object.position.set((index % W - (W - 1) / 2) * this.cellWidth, height / 2, ((H - 1) / 2 - Math.floor(index / W)) * this.cellDepth);
        object.scale.set(1, height, 1);
        object.updateMatrix();
        this.surface.setMatrixAt(index, object.matrix);
        color.copy(fromColors[index]).lerp(targetColors[index], eased);
        this.currentColors[index].copy(color);
        this.surface.setColorAt(index, color);
      }
      this.surface.instanceMatrix.needsUpdate = true;
      if (this.surface.instanceColor) this.surface.instanceColor.needsUpdate = true;
      this.render();
    };
    const animate = now => {
      const progress = duration ? Math.min(1, (now - start) / duration) : 1;
      apply(progress);
      if (progress < 1) this.transitionFrame = requestAnimationFrame(animate);
      else this.transitionFrame = 0;
    };
    animate(start);
  };

  ThreeMapView.prototype.focusRegion = function(regionKey) {
    const T = window.THREE;
    const region = REGIONS[regionKey];
    this.selectedRegion = regionKey;
    const [lat, lon] = region.center;
    const center = worldPoint(lon, lat);
    const nextTarget = new T.Vector3(center.x, 0.42, center.z);
    const offset = this.camera.position.clone().sub(this.target).normalize();
    const radius = regionKey === "all" ? 14.8 : 10.3;
    const nextPosition = nextTarget.clone().add(offset.multiplyScalar(radius));
    this.showRegionBounds(regionKey);
    if (this.cameraFrame) cancelAnimationFrame(this.cameraFrame);
    const fromPosition = this.camera.position.clone();
    const fromTarget = this.target.clone();
    const start = performance.now();
    const duration = prefersReducedMotion ? 0 : 420;
    const animate = now => {
      const t = duration ? Math.min(1, (now - start) / duration) : 1;
      const eased = 1 - (1 - t) ** 3;
      this.camera.position.lerpVectors(fromPosition, nextPosition, eased);
      this.target.lerpVectors(fromTarget, nextTarget, eased);
      this.camera.lookAt(this.target);
      this.render();
      if (t < 1) this.cameraFrame = requestAnimationFrame(animate);
      else this.cameraFrame = 0;
    };
    animate(start);
  };

  ThreeMapView.prototype.showRegionBounds = function(regionKey) {
    const T = window.THREE;
    if (regionKey === "all") { this.regionOutline.visible = false; return; }
    const region = REGIONS[regionKey];
    const nw = worldPoint(region.lonMin, region.latMax);
    const ne = worldPoint(region.lonMax, region.latMax);
    const se = worldPoint(region.lonMax, region.latMin);
    const sw = worldPoint(region.lonMin, region.latMin);
    const y = 1.74;
    const points = [nw, ne, se, sw].map(point => new T.Vector3(point.x, y, point.z));
    this.regionOutline.geometry.dispose();
    this.regionOutline.geometry = new T.BufferGeometry().setFromPoints(points);
    this.regionOutline.visible = true;
  };

  ThreeMapView.prototype.tick = function(delta) {
    if (prefersReducedMotion || !this.particles) return;
    const positions = this.particles.positions;
    for (let i = 0; i < this.particles.count; i++) {
      const offset = i * 6;
      positions[offset + 1] -= this.particles.speeds[i] * delta;
      positions[offset + 4] -= this.particles.speeds[i] * delta;
      if (positions[offset + 1] < 0.6) {
        const resetY = 2.5 + ((i * 29) % 100) / 100 * 1.5;
        positions[offset + 1] = resetY;
        positions[offset + 4] = resetY - (0.12 + ((i * 13) % 20) / 100);
      }
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
    this.render();
  };

  function initMapViews() {
    if (!window.THREE || !window.WebGL2RenderingContext) {
      ui.mapInstructions.textContent = "2D fallback active · WebGL2 is unavailable. Forecast controls, hover values and map-based shocks remain enabled.";
      return;
    }
    const entries = [["base", ui.base], ["shock", ui.shock], ["delta", ui.delta]];
    for (let index = 0; index < entries.length; index++) {
      const [key, canvas] = entries[index];
      try {
        mapViews[key] = new ThreeMapView(canvas, index);
        mapViews[key].onPaint = paintCellAt;
        mapViews[key].onHover = details => showHover(key, details);
        canvas.classList.add("three-map");
      } catch (error) {
        console.warn("Three.js map initialization failed; using 2D maps.", error);
        mapViews[key] = null;
      }
    }
    if (!Object.values(mapViews).some(Boolean)) {
      ui.mapInstructions.textContent = "2D fallback active · the 3D renderer could not start. Forecast controls and shocks remain enabled.";
      return;
    }
    if (Object.values(mapViews).some(view => !view)) ui.mapInstructions.textContent = "Some maps are using the 2D fallback. Forecast controls and shocks remain enabled.";
    let lastTime = performance.now();
    const animateAtmosphere = now => {
      const delta = Math.min((now - lastTime) / 1000, 0.06);
      lastTime = now;
      if (document.visibilityState === "visible") Object.values(mapViews).forEach(view => view?.tick(delta));
      if (!prefersReducedMotion) requestAnimationFrame(animateAtmosphere);
    };
    if (!prefersReducedMotion) requestAnimationFrame(animateAtmosphere);
  }

  function formatValue(value) {
    if (!Number.isFinite(value)) return "—";
    const magnitude = Math.abs(value);
    return magnitude >= 1000 ? value.toExponential(1) : value.toFixed(magnitude < 10 ? 1 : 0);
  }

  function showHover(key, details) {
    const target = document.querySelector(`#hover-${key}`);
    if (!details) { target.textContent = "Hover a cell to inspect"; return; }
    const lat = details.lat.toFixed(1);
    const lon = details.lon.toFixed(1);
    const value = formatValue(details.value);
    const unit = VARIABLES[ui.variable.value].unit;
    target.textContent = `${lat}°N · ${lon}°E    ${value} ${unit}`;
  }

  function activeDay() { return Number(ui.day.value) - 1; }

  function updateLegends(low, high, deltaSpan) {
    document.querySelector("#base-min").textContent = formatValue(low);
    document.querySelector("#base-max").textContent = formatValue(high);
    document.querySelector("#shock-min").textContent = formatValue(low);
    document.querySelector("#shock-max").textContent = formatValue(high);
    document.querySelector("#delta-min").textContent = `−${formatValue(deltaSpan)}`;
    document.querySelector("#delta-max").textContent = `+${formatValue(deltaSpan)}`;
  }

  function updateSummary(day, shockFrame, deltaField) {
    const region = REGIONS[activeRegion];
    let peakRain = -Infinity;
    let maxDelta = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (!regionMatches(activeRegion, i, j)) continue;
        peakRain = Math.max(peakRain, shockFrame[CHANNELS.tp][i][j]);
        maxDelta = Math.max(maxDelta, Math.abs(deltaField[i][j]));
      }
    }
    const risks = computeRegionRisks(activeRegion, day);
    ui.summaryRain.textContent = Number.isFinite(peakRain) ? formatValue(peakRain) : "—";
    ui.summaryHeat.textContent = risks.heat.toLocaleString("en-US");
    ui.summaryDelta.textContent = formatValue(maxDelta);
    ui.summaryRegion.textContent = region.label;
    ui.summaryRegionCells.textContent = activeRegion === "all" ? "32 × 36 model grid" : `${countRegionCells(activeRegion)} modeled cells`;
  }

  function countRegionCells(regionKey) {
    let total = 0;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) if (regionMatches(regionKey, i, j)) total++;
    return total;
  }

  function computeRegionRisks(regionKey, day) {
    if (!shocked) return { heat: 0, heavy: 0, cloud: 0, drought: 0 };
    const frame = shocked[day];
    const risks = { heat: 0, heavy: 0, cloud: 0, drought: 0 };
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (!regionMatches(regionKey, i, j)) continue;
        if (frame[CHANNELS.t2m][i][j] > 313.15) risks.heat++;
        const rain = frame[CHANNELS.tp][i][j];
        if (rain > 64.5 && rain <= 115.5) risks.heavy++;
        if (rain > 115.5) risks.cloud++;
        let rainSum = 0;
        let tempSum = 0;
        for (let step = 0; step < 10; step++) {
          rainSum += shocked[step][CHANNELS.tp][i][j];
          tempSum += shocked[step][CHANNELS.t2m][i][j];
        }
        if (rainSum < 25 && tempSum / 10 > 308.15) risks.drought++;
      }
    }
    return risks;
  }

  function updateRegionalSummary(day) {
    const summaries = INDIA_REGIONS.map(key => {
      const region = REGIONS[key];
      const risks = computeRegionRisks(key, day);
      const signal = [
        ["HEAT", risks.heat], ["HEAVY RAIN", risks.heavy], ["CLOUDBURST", risks.cloud], ["DROUGHT", risks.drought],
      ].sort((a, b) => b[1] - a[1])[0];
      const active = key === activeRegion ? " active" : "";
      return `<button class="region-summary-card${active}" type="button" data-region="${key}" aria-label="Focus ${region.label}, ${signal[1]} cells flagged for ${signal[0].toLowerCase()}"><strong>${region.label}</strong><span class="region-summary-value"><span>${signal[0]}</span><b>${signal[1]}</b></span></button>`;
    }).join("");
    ui.regionalSummary.innerHTML = summaries;
    ui.regionalStatus.textContent = `DAY ${String(day + 1).padStart(2, "0")} · SHOCK BRANCH`;
    ui.regionalStatus.classList.add("ready");
  }

  function updateImpacts(day) {
    if (!shocked) return;
    const risks = computeRegionRisks(activeRegion, day);
    document.querySelector("#heat-count").textContent = risks.heat.toLocaleString("en-US");
    document.querySelector("#rain-count").textContent = risks.heavy.toLocaleString("en-US");
    document.querySelector("#cloud-count").textContent = risks.cloud.toLocaleString("en-US");
    document.querySelector("#drought-count").textContent = risks.drought.toLocaleString("en-US");
    ui.activeRegionName.textContent = REGIONS[activeRegion].label;
    updateRegionalSummary(day);
  }

  function drawAll() {
    const day = activeDay();
    ui.dayLabel.innerHTML = `DAY ${String(day + 1).padStart(2, "0")} <span>·</span> 10`;
    const baseField = selectedField(baseline, day);
    const shockField = selectedField(shocked, day);
    if (!baseField || !shockField) return;
    const deltaField = baseField.map((row, i) => row.map((value, j) => shockField[i][j] - value));
    const [low, high] = mapRange(baseField, shockField);
    const deltaSpan = Math.max(0.1, ...deltaField.flat().map(Math.abs));
    if (mapViews.base) mapViews.base.update(baseField, [low, high]);
    else drawField(ui.base, baseField, [low, high]);
    if (mapViews.shock) mapViews.shock.update(shockField, [low, high]);
    else drawField(ui.shock, shockField, [low, high]);
    if (mapViews.delta) mapViews.delta.update(deltaField, deltaSpan, true);
    else drawField(ui.delta, deltaField, deltaSpan, true);
    updateLegends(low, high, deltaSpan);
    updateImpacts(day);
    updateSummary(day, shocked[day], deltaField);
  }

  function setShockStatus() {
    const labels = {
      none: "No shock applied. Click a map cell or choose a preset to create a local perturbation.",
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
    if (name === "warm") {
      ui.variable.value = "t2m";
      paintRegion({ variable: "t2m", value: 2, latMin: 8, latMax: 20, lonMin: 82, lonMax: 94 });
    }
    if (name === "arabian") {
      ui.variable.value = "wind";
      paintRegion({ variable: "wind", value: -3, latMin: 8, latMax: 20, lonMin: 65, lonMax: 75 });
    }
    if (name === "el-nino") {
      ui.variable.value = "t2m";
      paintRegion({ variable: "t2m", value: 2, latMin: 16, latMax: 27, lonMin: 72, lonMax: 86 });
    }
    configureUnit();
    setShockStatus();
    runForecast();
  }

  function paintCellAt(i, j) {
    if (i < 0 || i >= H || j < 0 || j >= W) return;
    const amount = Number(ui.shockSize.value);
    if (!Number.isFinite(amount) || amount === 0) return;
    const variable = ui.variable.value;
    const channels = variable === "wind" ? [CHANNELS.u10, CHANNELS.v10] : [CHANNELS[variable]];
    for (const channel of channels) perturbation[channel][i][j] += amount;
    activePreset = "custom";
    setShockStatus();
    ui.shockStatus.textContent += ` Cell ${6.5 + i}°N, ${65.5 + j}°E.`;
    window.clearTimeout(repaintTimer);
    repaintTimer = window.setTimeout(runForecast, 320);
  }

  function paintCellFallback(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const j = Math.floor((event.clientX - rect.left) / rect.width * W);
    const displayRow = Math.floor((event.clientY - rect.top) / rect.height * H);
    paintCellAt(H - 1 - displayRow, j);
  }

  function hoverFallback(canvas, event, key) {
    const rect = canvas.getBoundingClientRect();
    const j = Math.floor((event.clientX - rect.left) / rect.width * W);
    const displayRow = Math.floor((event.clientY - rect.top) / rect.height * H);
    const i = H - 1 - displayRow;
    const day = activeDay();
    const field = key === "base" ? selectedField(baseline, day) : selectedField(shocked, day);
    if (!field || i < 0 || i >= H || j < 0 || j >= W) return showHover(key, null);
    if (key === "delta") {
      const baseField = selectedField(baseline, day);
      showHover(key, { lat: 6.5 + i, lon: 65.5 + j, value: field[i][j] - baseField[i][j] });
    } else showHover(key, { lat: 6.5 + i, lon: 65.5 + j, value: field[i][j] });
  }

  function clearShock() {
    perturbation = zeroShock();
    activePreset = "none";
    setShockStatus();
    runForecast();
  }

  function selectRegion(regionKey) {
    if (!REGIONS[regionKey]) return;
    activeRegion = regionKey;
    ui.region.value = regionKey;
    Object.values(mapViews).forEach(view => view?.focusRegion(regionKey));
    if (shocked) drawAll();
    else {
      ui.summaryRegion.textContent = REGIONS[regionKey].label;
      ui.summaryRegionCells.textContent = `${countRegionCells(regionKey)} modeled cells`;
      ui.activeRegionName.textContent = REGIONS[regionKey].label;
    }
  }

  initMapViews();
  for (const [key, canvas] of [["base", ui.base], ["shock", ui.shock], ["delta", ui.delta]]) {
    if (!mapViews[key]) {
      canvas.addEventListener("click", event => paintCellFallback(canvas, event));
      canvas.addEventListener("mousemove", event => hoverFallback(canvas, event, key));
      canvas.addEventListener("mouseleave", () => showHover(key, null));
    }
  }
  document.querySelectorAll(".preset-button").forEach(button => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  document.querySelector("#clear-shock").addEventListener("click", clearShock);
  ui.variable.addEventListener("change", () => { configureUnit(); drawAll(); });
  ui.region.addEventListener("change", () => selectRegion(ui.region.value));
  ui.day.addEventListener("input", drawAll);
  ui.refresh.addEventListener("click", runForecast);
  ui.regionalSummary.addEventListener("click", event => {
    const button = event.target.closest("[data-region]");
    if (button) selectRegion(button.dataset.region);
  });
  ui.play.addEventListener("click", () => {
    if (playTimer) {
      window.clearInterval(playTimer);
      playTimer = null;
      ui.play.textContent = "▶";
      ui.play.setAttribute("aria-label", "Play forecast timeline");
      ui.play.classList.remove("playing");
      return;
    }
    ui.play.textContent = "Ⅱ";
    ui.play.setAttribute("aria-label", "Pause forecast timeline");
    ui.play.classList.add("playing");
    playTimer = window.setInterval(() => {
      ui.day.value = Number(ui.day.value) >= 10 ? 1 : Number(ui.day.value) + 1;
      drawAll();
    }, 1000);
  });

  configureUnit();
  selectRegion("all");
  runForecast();
})();
