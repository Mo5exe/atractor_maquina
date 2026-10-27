/*
 * "Lo que ve la máquina": detector de objetos YOLO (YOLOX-nano, 80 clases de COCO)
 * corriendo en el navegador con ONNX Runtime Web, sobre el video de la cámara.
 *
 * Uso:
 *   import { ObjectDetector } from "./objects.js";
 *   const det = new ObjectDetector({ video, getMirror, getMinConf, isWanted, onObjects, onStatus });
 *   det.start();
 *
 * onObjects([{ id, cls, label, conf, x, y, w, h, color }])
 *   x, y, w, h normalizados 0..1 (rectángulo, ya espejado si corresponde)
 *   conf: la "confianza" de la máquina (0..1) · color: color promedio del objeto en la cámara
 *
 * El modelo YOLOX es de Megvii (licencia Apache 2.0) y está en /yolo/yolox_nano.onnx.
 */

// "Ojos" de la máquina: más grande = ve mejor, pero más lento.
// "yolox": 80 cosas (COCO). "v8": 601 cosas (Open Images V7, YOLOv8 de Ultralytics).
export const MODELS = {
  fast: { url: "/yolo/yolox_nano.onnx", size: 416, label: "rápido", kind: "yolox" },
  medium: { url: "/yolo/yolox_tiny.onnx", size: 416, label: "intermedio", kind: "yolox" },
  precise: { url: "/yolo/yolox_s.onnx", size: 640, label: "preciso", kind: "yolox" },
  many: { url: "/yolo/yolov8n-oiv7.onnx", size: 640, label: "601 cosas", kind: "v8" },
  manyPrecise: { url: "/yolo/yolov8s-oiv7.onnx", size: 640, label: "601 cosas, preciso", kind: "v8" }
};

// Las 80 clases de COCO, en castellano (rioplatense).
export const LABELS_ES = [
  "persona", "bicicleta", "auto", "moto", "avión", "colectivo", "tren", "camión", "barco", "semáforo",
  "hidrante", "cartel de pare", "parquímetro", "banco", "pájaro", "gato", "perro", "caballo", "oveja", "vaca",
  "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas", "cartera", "corbata", "valija", "frisbee",
  "esquís", "snowboard", "pelota", "barrilete", "bate", "guante", "skate", "tabla de surf", "raqueta", "botella",
  "copa", "taza", "tenedor", "cuchillo", "cuchara", "bol", "banana", "manzana", "sándwich", "naranja",
  "brócoli", "zanahoria", "pancho", "pizza", "dona", "torta", "silla", "sillón", "planta", "cama",
  "mesa", "inodoro", "televisor", "notebook", "mouse", "control remoto", "teclado", "celular", "microondas", "horno",
  "tostadora", "pileta", "heladera", "libro", "reloj", "florero", "tijera", "osito de peluche", "secador de pelo", "cepillo de dientes"
];

let ortPromise = null;
function loadOrt() {
  if (!ortPromise) {
    ortPromise = import("/ort/ort.min.mjs").then((ort) => {
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1; // sin hilos: anda en cualquier servidor (también Render)
      return ort;
    });
  }
  return ortPromise;
}

// Grilla de YOLOX para decodificar las salidas (strides 8, 16 y 32).
import { LABELS_OIV7 } from "./labels-oiv7.js";

const GRIDS = {};
function gridFor(size) {
  if (GRIDS[size]) return GRIDS[size];
  const gx = [], gy = [], st = [];
  for (const s of [8, 16, 32]) {
    const n = size / s;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { gx.push(x); gy.push(y); st.push(s); }
  }
  return (GRIDS[size] = { gx, gy, st, n: st.length });
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

export class ObjectDetector {
  constructor(opts) {
    this.video = opts.video;
    this.getMirror = opts.getMirror || (() => true);
    this.getMinConf = opts.getMinConf || (() => 0.25);
    this.isWanted = opts.isWanted || (() => true);
    this.getModel = opts.getModel || (() => "fast");
    this.onObjects = opts.onObjects || (() => {});
    this.onStatus = opts.onStatus || (() => {});
    this.session = null;
    this.running = false;
    this.tracks = [];
    this.nextId = 1;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.modelKey = null;
    this.loading = null;
    this._tick = this._tick.bind(this);
  }

  // Carga (o cambia) el modelo según el "ojo" elegido.
  async _load(key) {
    const m = MODELS[key] || MODELS.fast;
    this.onStatus("Cargando YOLO " + m.label + " (lo que ve la máquina)…", "info");
    const ort = await loadOrt();
    this.ort = ort;
    const session = await ort.InferenceSession.create(m.url, { executionProviders: ["wasm"] });
    this.session = session;
    this.modelKey = key;
    this.kind = m.kind;
    this.labels = m.kind === "v8" ? LABELS_OIV7 : LABELS_ES;
    this.size = m.size;
    this.canvas.width = m.size;
    this.canvas.height = m.size;
    this.input = new Float32Array(3 * m.size * m.size);
    this.tracks = [];
    this.onStatus("YOLO " + m.label + " listo: la máquina está mirando.", "ok");
  }

  async start() {
    if (this.running) return true;
    this.running = true;
    try {
      const key = MODELS[this.getModel()] ? this.getModel() : "fast";
      if (!this.session || this.modelKey !== key) await this._load(key);
    } catch (err) {
      console.error("No se pudo cargar YOLO", err);
      this.onStatus("No se pudo cargar YOLO: " + (err && err.message ? err.message : err), "error");
      this.running = false;
      return false;
    }
    setTimeout(this._tick, 0);
    return true;
  }

  stop() {
    this.running = false;
    this.tracks = [];
    this.onObjects([]);
  }

  async _tick() {
    if (!this.running) return;
    const t0 = performance.now();
    let wait = 120;
    const want = MODELS[this.getModel()] ? this.getModel() : "fast";
    if (this.isWanted() && want !== this.modelKey) {
      try { await this._load(want); } catch (err) { console.error(err); this.onStatus("No se pudo cargar ese YOLO.", "error"); }
    }
    if (!this.isWanted()) {
      if (this.tracks.length) { this.tracks = []; this.onObjects([]); }
      wait = 500;
    } else if (this.video && this.video.readyState >= 2 && this.video.videoWidth) {
      try {
        const dets = await this._detect();
        this._track(dets);
        this.onObjects(this.tracks.filter((t) => t.seen > 0).map((t) => this._public(t)));
      } catch (err) {
        console.error("Error de YOLO", err);
      }
      // Darle respiro a la compu: esperar más o menos lo que tardó.
      wait = Math.max(40, Math.min(250, performance.now() - t0));
    }
    setTimeout(this._tick, wait);
  }

  async _detect() {
    const v = this.video;
    const SIZE = this.size;
    const GRID = gridFor(SIZE);
    const W = v.videoWidth, H = v.videoHeight;
    const r = Math.min(SIZE / W, SIZE / H);
    const rw = Math.round(W * r), rh = Math.round(H * r);
    const c = this.ctx;
    c.fillStyle = "rgb(114,114,114)";
    c.fillRect(0, 0, SIZE, SIZE);
    c.drawImage(v, 0, 0, rw, rh);
    const px = c.getImageData(0, 0, SIZE, SIZE).data;
    this.pixels = px; // para sacar el color de cada objeto
    const plane = SIZE * SIZE;
    const inp = this.input;
    if (this.kind === "v8") {
      // YOLOv8: RGB de 0 a 1
      for (let i = 0, p = 0; i < plane; i++, p += 4) {
        inp[i] = px[p] / 255;
        inp[plane + i] = px[p + 1] / 255;
        inp[2 * plane + i] = px[p + 2] / 255;
      }
    } else {
      // YOLOX: BGR de 0 a 255
      for (let i = 0, p = 0; i < plane; i++, p += 4) {
        inp[i] = px[p + 2];
        inp[plane + i] = px[p + 1];
        inp[2 * plane + i] = px[p];
      }
    }
    const tensor = new this.ort.Tensor("float32", inp, [1, 3, SIZE, SIZE]);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const data = out[this.session.outputNames[0]].data;
    const minConf = this.getMinConf();
    const cand = [];
    if (this.kind === "v8") {
      // YOLOv8: [4 + clases, N] → cx, cy, w, h en píxeles del cuadro y una nota por clase.
      const dims = out[this.session.outputNames[0]].dims;
      const C = dims[1] - 4, N = dims[2];
      for (let i = 0; i < N; i++) {
        let best = 0, cls = 0;
        for (let k = 0; k < C; k++) {
          const v = data[(4 + k) * N + i];
          if (v > best) { best = v; cls = k; }
        }
        if (best < minConf) continue;
        const cx = data[i], cy = data[N + i], bw = data[2 * N + i], bh = data[3 * N + i];
        cand.push({ cls, conf: best, x: (cx - bw / 2) / rw, y: (cy - bh / 2) / rh, w: bw / rw, h: bh / rh, px: [cx, cy, bw, bh] });
      }
    }
    for (let i = 0; this.kind !== "v8" && i < GRID.n; i++) {
      const o = i * 85;
      const obj = data[o + 4];
      if (obj < minConf * 0.5) continue;
      let best = 0, cls = 0;
      for (let k = 0; k < 80; k++) {
        const s = data[o + 5 + k];
        if (s > best) { best = s; cls = k; }
      }
      const conf = obj * best;
      if (conf < minConf) continue;
      const s = GRID.st[i];
      const cx = (data[o] + GRID.gx[i]) * s;
      const cy = (data[o + 1] + GRID.gy[i]) * s;
      const bw = Math.exp(data[o + 2]) * s;
      const bh = Math.exp(data[o + 3]) * s;
      cand.push({ cls, conf, x: (cx - bw / 2) / rw, y: (cy - bh / 2) / rh, w: bw / rw, h: bh / rh, px: [cx, cy, bw, bh] });
    }
    // NMS por clase.
    cand.sort((a, b) => b.conf - a.conf);
    const keep = [];
    for (const d of cand) {
      if (keep.length >= 20) break;
      if (keep.some((k) => k.cls === d.cls && iou(k, d) > 0.45)) continue;
      keep.push(d);
    }
    const mirror = this.getMirror();
    for (const d of keep) {
      d.x = Math.max(0, d.x); d.y = Math.max(0, d.y);
      d.w = Math.min(1 - d.x, d.w); d.h = Math.min(1 - d.y, d.h);
      if (mirror) d.x = 1 - d.x - d.w;
      d.color = this._color(d.px);
    }
    return keep;
  }

  // Color promedio del centro del objeto (lo que la cámara ve de esa cosa).
  _color(b) {
    const px = this.pixels;
    const SIZE = this.size;
    let r = 0, g = 0, bl = 0, n = 0;
    const x0 = Math.max(0, Math.round(b[0] - b[2] * 0.25)), x1 = Math.min(SIZE - 1, Math.round(b[0] + b[2] * 0.25));
    const y0 = Math.max(0, Math.round(b[1] - b[3] * 0.25)), y1 = Math.min(SIZE - 1, Math.round(b[1] + b[3] * 0.25));
    const step = Math.max(1, Math.round((x1 - x0) / 12));
    for (let y = y0; y <= y1; y += step) {
      for (let x = x0; x <= x1; x += step) {
        const p = (y * SIZE + x) * 4;
        r += px[p]; g += px[p + 1]; bl += px[p + 2]; n++;
      }
    }
    if (!n) return "#ffffff";
    const hex = (v) => Math.round(v / n).toString(16).padStart(2, "0");
    return "#" + hex(r) + hex(g) + hex(bl);
  }

  // Seguimiento: cada cosa conserva su número entre cuadro y cuadro, y se suaviza.
  _track(dets) {
    const used = new Set();
    for (const t of this.tracks) {
      let best = null, bestIou = 0.2;
      dets.forEach((d, i) => {
        if (used.has(i) || d.cls !== t.cls) return;
        const v = iou(t, d);
        if (v > bestIou) { bestIou = v; best = i; }
      });
      if (best !== null) {
        used.add(best);
        const d = dets[best];
        const k = 0.5;
        t.x += (d.x - t.x) * k; t.y += (d.y - t.y) * k; t.w += (d.w - t.w) * k; t.h += (d.h - t.h) * k;
        t.conf += (d.conf - t.conf) * 0.35;
        t.color = d.color;
        t.seen = 3; // vidas: aguanta unos cuadros sin verse
      } else {
        t.seen--;
        t.conf *= 0.8;
      }
    }
    this.tracks = this.tracks.filter((t) => t.seen > -1);
    dets.forEach((d, i) => {
      if (used.has(i)) return;
      this.tracks.push({ id: this.nextId++, cls: d.cls, conf: d.conf, x: d.x, y: d.y, w: d.w, h: d.h, color: d.color, seen: 3 });
    });
  }

  _public(t) {
    const r = (v) => Math.round(v * 1000) / 1000;
    return { id: t.id, cls: t.cls, label: (this.labels || LABELS_ES)[t.cls] || "cosa", conf: r(t.conf), x: r(t.x), y: r(t.y), w: r(t.w), h: r(t.h), color: t.color };
  }
}
