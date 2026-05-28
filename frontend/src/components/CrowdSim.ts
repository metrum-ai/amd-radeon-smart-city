// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

export interface CrowdSimConfig {
  people: number;
  heat: number;
  heatX: number;
  heatY: number;
  sevColor: [number, number, number];
}

interface Person {
  x: number;
  y: number;
  w: number;
  h: number;
  vx: number;
  vy: number;
  shade: number;
  skin: boolean;
}

interface BBox {
  person: number;
  alert: boolean;
}

export class CrowdSim {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  cfg: CrowdSimConfig;
  people: Person[] = [];
  bboxes: BBox[] = [];
  frame = 0;

  constructor(canvas: HTMLCanvasElement, cfg: CrowdSimConfig) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.cfg = cfg;
    this._initPeople();
    this._initBboxes();
  }

  private _initPeople() {
    for (let i = 0; i < this.cfg.people; i++) {
      this.people.push({
        x: Math.random(),
        y: 0.35 + Math.random() * 0.6,
        w: 0.015 + Math.random() * 0.01,
        h: 0.06 + Math.random() * 0.04,
        vx: (Math.random() - 0.5) * 0.0008,
        vy: (Math.random() - 0.5) * 0.0004,
        shade: 40 + Math.random() * 80,
        skin: Math.random() > 0.5,
      });
    }
  }

  private _initBboxes() {
    const n = Math.min(8, Math.floor(this.cfg.people / 6));
    for (let i = 0; i < n; i++) {
      this.bboxes.push({ person: i * 6, alert: i < 2 });
    }
  }

  resize() {
    const r = this.canvas.parentElement?.getBoundingClientRect();
    if (!r) return;
    this.canvas.width = Math.floor(r.width);
    this.canvas.height = Math.floor(r.height);
  }

  draw() {
    const { ctx, cfg, people, bboxes } = this;
    const W = this.canvas.width;
    const H = this.canvas.height;
    if (!W || !H) return;
    this.frame++;

    ctx.fillStyle = "#0a0c12";
    ctx.fillRect(0, 0, W, H);

    const grd = ctx.createLinearGradient(0, H * 0.3, 0, H);
    grd.addColorStop(0, "#0e1118");
    grd.addColorStop(0.5, "#161c24");
    grd.addColorStop(1, "#1a2030");
    ctx.fillStyle = grd;
    ctx.fillRect(0, H * 0.3, W, H * 0.7);

    for (const p of people) {
      p.x += p.vx + Math.sin(this.frame * 0.02 + p.y * 10) * 0.0003;
      p.y += p.vy + Math.cos(this.frame * 0.015 + p.x * 8) * 0.0002;
      if (p.x < -0.02) p.x = 1.02;
      if (p.x > 1.02) p.x = -0.02;
      if (p.y < 0.35) p.y = 0.35;
      if (p.y > 0.95) p.y = 0.95;

      const px = p.x * W;
      const py = p.y * H;
      const pw = p.w * W;
      const ph = p.h * H;
      const s = p.shade;

      ctx.fillStyle = p.skin
        ? `rgb(${s + 60},${s + 30},${s})`
        : `rgb(${s},${s},${s + 10})`;
      ctx.fillRect(px - pw / 2, py - ph, pw, ph);
      ctx.beginPath();
      ctx.arc(px, py - ph - pw * 0.6, pw * 0.7, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${s + 80},${s + 50},${s + 20})`;
      ctx.fill();
    }

    const [r, g, b] = cfg.sevColor;
    const hg = ctx.createRadialGradient(
      cfg.heatX * W,
      cfg.heatY * H,
      0,
      cfg.heatX * W,
      cfg.heatY * H,
      Math.max(W, H) * 0.5,
    );
    hg.addColorStop(0, `rgba(${r},${g},${b},${cfg.heat * 0.6})`);
    hg.addColorStop(
      0.3,
      `rgba(${r},${Math.min(255, g + 80)},0,${cfg.heat * 0.35})`,
    );
    hg.addColorStop(0.6, `rgba(${r},${g},${b},${cfg.heat * 0.1})`);
    hg.addColorStop(1, "transparent");
    ctx.fillStyle = hg;
    ctx.fillRect(0, 0, W, H);

    if (cfg.heat > 0.3) {
      const hg2 = ctx.createRadialGradient(
        (cfg.heatX + 0.25) * W,
        (cfg.heatY + 0.15) * H,
        0,
        (cfg.heatX + 0.25) * W,
        (cfg.heatY + 0.15) * H,
        W * 0.3,
      );
      hg2.addColorStop(0, `rgba(255,200,0,${cfg.heat * 0.35})`);
      hg2.addColorStop(0.5, `rgba(${r},${g},${b},${cfg.heat * 0.15})`);
      hg2.addColorStop(1, "transparent");
      ctx.fillStyle = hg2;
      ctx.fillRect(0, 0, W, H);
    }

    for (const bb of bboxes) {
      const p = people[bb.person];
      if (!p) continue;
      const px = p.x * W;
      const py = p.y * H;
      const pw = p.w * W * 2.5;
      const ph = p.h * H * 1.6;
      ctx.strokeStyle = bb.alert ? "rgba(237,28,36,.8)" : "rgba(0,124,151,.6)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px - pw / 2, py - ph + 2, pw, ph);
      ctx.fillStyle = bb.alert ? "rgba(237,28,36,.7)" : "rgba(0,124,151,.5)";
      ctx.fillRect(px - pw / 2, py - ph - 1, 28, 10);
      ctx.fillStyle = "#fff";
      ctx.font = "7px monospace";
      ctx.fillText(bb.alert ? "0.94" : "0.87", px - pw / 2 + 2, py - ph + 7);
    }
  }
}
