/**
 * E-Brain 315M Training Orchestrator — Cloudflare Worker
 *
 * Endpoints:
 *   POST /heartbeat      (auth: X-Token: HEARTBEAT_TOKEN)
 *   GET  /status         (public JSON)
 *   GET  /control        (public JSON — polled by training kernel)
 *   POST /control        (auth: X-Token: DASHBOARD_TOKEN)
 *   GET  /               (dashboard HTML)
 *   POST /telegram_test  (auth: X-Token: DASHBOARD_TOKEN)
 *
 * KV namespace : EBRAIN_KV
 * Secrets      : HEARTBEAT_TOKEN, DASHBOARD_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
 */

const TOTAL_STEPS = 194000;
const MODEL_PARAMS = "315M";
const ARCH        = "D=1024 L=24 H=16 T=4096 · 16K BPE · FlashAttention";
const TARGET_TOKENS = "6.36B";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
}

function htmlRes(body, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}

function requireHeader(request, env, headerName, envVar) {
  const token = request.headers.get(headerName);
  if (!token || token !== env[envVar]) return new Response("Unauthorized", { status: 401 });
  return null;
}

async function sendTelegram(env, message) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID) return;
  try {
    await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text: `E-Brain 100M\n${message}` }),
    });
  } catch (_) {}
}

async function appendAlert(env, message) {
  let log = [];
  try { log = JSON.parse(await env.EBRAIN_KV.get("alert_log") || "[]"); } catch (_) {}
  log.unshift(`[${new Date().toISOString()}] ${message}`);
  if (log.length > 50) log = log.slice(0, 50);
  await env.EBRAIN_KV.put("alert_log", JSON.stringify(log));
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async function handleHeartbeat(request, env) {
  const authErr = requireHeader(request, env, "X-Token", "HEARTBEAT_TOKEN");
  if (authErr) return authErr;

  let payload;
  try { payload = await request.json(); }
  catch (_) { return json({ error: "invalid JSON" }, 400); }

  const required = ["step", "loss", "lr", "tok_s", "vram_gb", "eta_h"];
  for (const f of required) {
    if (payload[f] === undefined) return json({ error: `missing field: ${f}` }, 400);
  }

  payload.received_at = new Date().toISOString();
  await env.EBRAIN_KV.put("last_heartbeat", JSON.stringify(payload));

  const alerts = [];
  if (payload.nan_alert)       alerts.push(`NaN detected at step ${payload.step}`);
  if (payload.loss_spike)      alerts.push(`Loss spike at step ${payload.step}: ${payload.loss.toFixed(4)}`);
  if (payload.val_alert)       alerts.push(`Val degraded at step ${payload.step}: ${payload.val_alert}`);
  if (payload.lang_gap_alert)  alerts.push(`Language gap >0.15 at step ${payload.step}: ${payload.lang_gap_alert}`);

  for (const a of alerts) {
    await appendAlert(env, a);
    await sendTelegram(env, a);
  }

  if (payload.val_per_lang) {
    let hist = [];
    try { hist = JSON.parse(await env.EBRAIN_KV.get("val_history") || "[]"); } catch (_) {}
    hist.push({ step: payload.step, ts: payload.received_at, bpb: payload.val_per_lang });
    if (hist.length > 500) hist = hist.slice(-500);
    await env.EBRAIN_KV.put("val_history", JSON.stringify(hist));
  }

  return json({ ok: true, received_at: payload.received_at });
}

async function handleGetStatus(env) {
  const hb      = JSON.parse(await env.EBRAIN_KV.get("last_heartbeat") || "{}");
  const flag    = await env.EBRAIN_KV.get("control_flag") || "RUN";
  const valHist = JSON.parse(await env.EBRAIN_KV.get("val_history")     || "[]");
  const sessions= JSON.parse(await env.EBRAIN_KV.get("session_history") || "[]");
  const alerts  = JSON.parse(await env.EBRAIN_KV.get("alert_log")       || "[]");
  const step    = hb.step || 0;

  return json({
    control_flag: flag,
    step, total_steps: TOTAL_STEPS,
    progress_pct: parseFloat(((step / TOTAL_STEPS) * 100).toFixed(2)),
    loss: hb.loss, lr: hb.lr, tok_s: hb.tok_s,
    vram_gb: hb.vram_gb, eta_h: hb.eta_h,
    last_heartbeat: hb.received_at,
    val_per_lang: hb.val_per_lang,
    checkpoint_hash: hb.checkpoint_hash,
    val_history: valHist.slice(-50),
    session_history: sessions,
    recent_alerts: alerts.slice(0, 20),
  });
}

async function handleGetControl(env) {
  const flag = await env.EBRAIN_KV.get("control_flag") || "RUN";
  return json({ control: flag });
}

async function handlePostControl(request, env) {
  const authErr = requireHeader(request, env, "X-Token", "DASHBOARD_TOKEN");
  if (authErr) return authErr;

  let body;
  try { body = await request.json(); }
  catch (_) { return json({ error: "invalid JSON" }, 400); }

  const { control } = body;
  if (!["RUN", "PAUSE", "STOP"].includes(control)) {
    return json({ error: "control must be RUN | PAUSE | STOP" }, 400);
  }

  await env.EBRAIN_KV.put("control_flag", control);
  await appendAlert(env, `Control set to ${control} via dashboard`);
  await sendTelegram(env, `Control set to ${control} via dashboard`);

  return json({ ok: true, control });
}

async function handleTelegramTest(request, env) {
  const authErr = requireHeader(request, env, "X-Token", "DASHBOARD_TOKEN");
  if (authErr) return authErr;
  await sendTelegram(env, "Test ping from E-Brain dashboard");
  return json({ ok: true });
}

// ─── Dashboard HTML ────────────────────────────────────────────────────────────

async function handleDashboard(env) {
  const hb       = JSON.parse(await env.EBRAIN_KV.get("last_heartbeat") || "{}");
  const flag     = await env.EBRAIN_KV.get("control_flag") || "RUN";
  const valHist  = JSON.parse(await env.EBRAIN_KV.get("val_history")     || "[]");
  const alerts   = JSON.parse(await env.EBRAIN_KV.get("alert_log")       || "[]");
  const sessions = JSON.parse(await env.EBRAIN_KV.get("session_history") || "[]");

  const step    = hb.step || 0;
  const pct     = ((step / TOTAL_STEPS) * 100).toFixed(2);
  const isAlive = !!hb.received_at;
  const staleSec= isAlive ? Math.floor((Date.now() - new Date(hb.received_at).getTime()) / 1000) : null;
  const isStale = staleSec !== null && staleSec > 15 * 60;

  // Derive status word
  let status = "IDLE";
  // accent-class drives the one accent color across status indicator + bar
  // values map to the CSS class: .s-training .s-paused .s-stopped .s-error .s-idle .s-done
  let statusClass = "s-idle";
  let statusLine  = "No active session. Trigger a run from GitHub Actions to begin.";

  if (hb.nan_alert) {
    status = "ERROR"; statusClass = "s-error";
    statusLine = `NaN detected at step ${step}. Training is unstable — pause and inspect.`;
  } else if (flag === "STOP") {
    status = "STOPPED"; statusClass = "s-stopped";
    statusLine = "Stop signal sent. Kernel is checkpointing and will exit.";
  } else if (isStale && flag === "RUN") {
    status = "STALLED"; statusClass = "s-paused";
    statusLine = "No heartbeat for more than 15 minutes. Kernel may have crashed.";
  } else if (flag === "PAUSE") {
    status = "PAUSED"; statusClass = "s-paused";
    statusLine = "Training is paused. Send Resume when ready to continue.";
  } else if (flag === "RUN" && isAlive && !isStale) {
    status = "TRAINING"; statusClass = "s-training";
    statusLine = `Step ${step.toLocaleString()} of ${TOTAL_STEPS.toLocaleString()} — kernel is live.`;
  } else if (step >= TOTAL_STEPS && step > 0) {
    status = "DONE"; statusClass = "s-done";
    statusLine = `All ${TOTAL_STEPS.toLocaleString()} steps complete.`;
  }

  // Heartbeat freshness
  let freshLabel = "Never";
  if (staleSec !== null) {
    if (staleSec < 60) freshLabel = `${staleSec}s ago`;
    else if (staleSec < 3600) freshLabel = `${Math.floor(staleSec / 60)}m ago`;
    else freshLabel = `${(staleSec / 3600).toFixed(1)}h ago`;
  }

  // Metric helpers
  const fmtF = (v, d = 4) => (v !== undefined && v !== null) ? Number(v).toFixed(d) : "—";
  const fmtI = v => (v !== undefined && v !== null) ? Math.round(v).toLocaleString() : "—";
  const fmtE = v => (v !== undefined && v !== null) ? Number(v).toExponential(2) : "—";

  // Loss sparkline (SVG, clean, no gradient fill — just the line)
  const sparkPts = valHist.slice(-24).map(v => {
    if (!v.bpb) return null;
    const vals = Object.values(v.bpb);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }).filter(x => x !== null);

  let sparkSVG = `<p class="empty-state">Validation loss data will appear here during training.</p>`;
  if (sparkPts.length >= 2) {
    const minV = Math.min(...sparkPts), maxV = Math.max(...sparkPts);
    const range = maxV - minV || 0.01;
    const W = 400, H = 64, px = 4;
    const coords = sparkPts.map((v, i) => {
      const x = px + (i / (sparkPts.length - 1)) * (W - 2 * px);
      const y = H - px - ((v - minV) / range) * (H - 2 * px);
      return [x, y];
    });
    const polyline = coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const lastBpb  = sparkPts[sparkPts.length - 1].toFixed(4);
    const firstBpb = sparkPts[0];
    const delta    = sparkPts[sparkPts.length - 1] - firstBpb;
    const trendTxt = Math.abs(delta) < 0.0005 ? "stable" : delta < 0 ? `down ${Math.abs(delta).toFixed(4)}` : `up ${delta.toFixed(4)}`;

    sparkSVG = `
      <div class="spark-meta">
        <span class="spark-current">${lastBpb} <span class="spark-unit">bpb</span></span>
        <span class="spark-trend ${delta < 0 ? "trend-good" : delta > 0 ? "trend-bad" : ""}">${trendTxt} over ${sparkPts.length} evals</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
        <polyline points="${polyline}" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
      </svg>`;
  }

  // Per-language BPB table rows
  const LANGS  = ["en","es","ru","fr","de","ja","zh","hi","bn","ar"];
  const LNAMES = { en:"English",es:"Spanish",ru:"Russian",fr:"French",de:"German",ja:"Japanese",zh:"Chinese",hi:"Hindi",bn:"Bengali",ar:"Arabic" };
  let langRows = `<tr><td colspan="3" class="empty-cell">No evaluation data yet.</td></tr>`;
  if (hb.val_per_lang) {
    langRows = LANGS.map(l => {
      const bpb = hb.val_per_lang[l];
      if (bpb === undefined) return "";
      let prev = null;
      if (valHist.length >= 2) prev = valHist[valHist.length - 2]?.bpb?.[l];
      const delta = prev !== null ? bpb - prev : null;
      const trendCell = delta === null ? `<td class="td-num muted">—</td>`
        : delta < -0.0001 ? `<td class="td-num trend-good">-${Math.abs(delta).toFixed(4)}</td>`
        : delta > 0.0001  ? `<td class="td-num trend-bad">+${delta.toFixed(4)}</td>`
        : `<td class="td-num muted">0</td>`;
      return `<tr><td>${LNAMES[l]}</td><td class="td-num">${bpb.toFixed(4)}</td>${trendCell}</tr>`;
    }).join("");
  }

  // Alert rows
  const alertRows = alerts.slice(0, 8).map(a => {
    const isWarn = a.includes("NaN") || a.includes("spike") || a.includes("gap") || a.includes("degraded");
    return `<div class="alert-item ${isWarn ? "alert-warn" : "alert-info"}">${a}</div>`;
  }).join("") || `<p class="empty-state">No alerts.</p>`;

  // Session history rows
  const sessRows = sessions.slice(-8).reverse().map(s =>
    `<tr>
      <td>${s.date || "—"}</td>
      <td class="td-num">${(s.steps || 0).toLocaleString()}</td>
      <td class="td-num">${s.tokens_B ? s.tokens_B.toFixed(3) + "B" : "—"}</td>
      <td class="td-num">${s.mean_val_bpb ? s.mean_val_bpb.toFixed(4) : "—"}</td>
    </tr>`
  ).join("") || `<tr><td colspan="4" class="empty-cell">No sessions recorded yet.</td></tr>`;

  const pctNum = parseFloat(pct);

  return htmlRes(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E-Brain — ${status}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Tokens ───────────────────────────────────────────────────────────────── */
:root {
  --hue: 220;

  /* Neutrals — cool, tinted, never pure gray */
  --gray-950: oklch(11%  0.010 var(--hue));
  --gray-900: oklch(16%  0.012 var(--hue));
  --gray-800: oklch(23%  0.012 var(--hue));
  --gray-700: oklch(32%  0.012 var(--hue));
  --gray-600: oklch(44%  0.012 var(--hue));
  --gray-500: oklch(57%  0.012 var(--hue));
  --gray-400: oklch(70%  0.010 var(--hue));
  --gray-300: oklch(82%  0.008 var(--hue));
  --gray-200: oklch(90%  0.006 var(--hue));
  --gray-100: oklch(95%  0.005 var(--hue));
  --gray-50:  oklch(98%  0.004 var(--hue));

  /* One accent — teal, not purple, not indigo */
  --accent-hue: 168;
  --accent:       oklch(56% 0.13 var(--accent-hue));
  --accent-hover: oklch(51% 0.13 var(--accent-hue));
  --accent-fg:    oklch(99% 0 0);
  --accent-subtle:oklch(23% 0.06 var(--accent-hue));

  /* Semantic roles */
  --bg:          var(--gray-950);
  --bg-raised:   var(--gray-900);
  --bg-elevated: var(--gray-800);
  --border:      rgba(255,255,255,0.07);
  --border-mid:  rgba(255,255,255,0.11);
  --text:        oklch(93% 0.006 var(--hue));
  --text-muted:  oklch(60% 0.010 var(--hue));
  --text-faint:  oklch(40% 0.010 var(--hue));

  /* Status — desaturated, never neon */
  --s-training: oklch(62% 0.14 148);   /* green */
  --s-paused:   oklch(72% 0.13 75);    /* amber */
  --s-stopped:  oklch(55% 0.17 22);    /* red */
  --s-error:    oklch(55% 0.17 22);
  --s-done:     oklch(60% 0.12 230);   /* blue */
  --s-idle:     var(--text-faint);

  /* Radius — 2 values only */
  --radius-sm: 6px;
  --radius-md: 10px;

  /* Shadows — low opacity, tinted */
  --shadow-xs: 0 1px 2px oklch(5% 0.01 var(--hue) / 0.4);
  --shadow-sm: 0 1px 2px oklch(5% 0.01 var(--hue) / 0.4),
               0 3px 8px oklch(5% 0.01 var(--hue) / 0.3);

  /* Motion */
  --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
  --dur-fast: 120ms;
  --dur-base: 200ms;

  /* Layout */
  --container: 1100px;
  --gutter: clamp(16px, 3vw, 32px);

  /* Spacing */
  --sp-1: 4px;  --sp-2: 8px;   --sp-3: 12px;  --sp-4: 16px;
  --sp-5: 24px; --sp-6: 32px;  --sp-7: 48px;  --sp-8: 64px;

  /* Type */
  --font-sans: "Geist", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "Geist Mono", ui-monospace, monospace;
  --text-xs:   12px;  --text-sm: 14px;  --text-base: 16px;
  --text-lg:   18px;  --text-xl: 20px;  --text-2xl: 24px;
  --text-hero: clamp(40px, 6vw + 16px, 80px);
}

/* ── Reset ────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}
::selection { background: var(--accent-subtle); color: var(--text); }

/* ── Layout shell ─────────────────────────────────────────────────────────── */
.shell {
  max-width: var(--container);
  margin: 0 auto;
  padding: var(--sp-6) var(--gutter) var(--sp-8);
}

/* ── Top bar ──────────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: var(--sp-6);
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--sp-6);
  gap: var(--sp-4);
  flex-wrap: wrap;
}
.brand {
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--text);
  letter-spacing: -0.01em;
}
.brand span { color: var(--text-muted); font-weight: 400; margin-left: var(--sp-2); }
.topbar-right {
  font-size: var(--text-xs);
  color: var(--text-faint);
  font-variant-numeric: tabular-nums;
}
.topbar-right strong { color: var(--text-muted); }

/* ── Status hero ──────────────────────────────────────────────────────────── */
.status-hero {
  padding: var(--sp-7) 0 var(--sp-6);
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--sp-6);
}
.status-row {
  display: flex;
  align-items: baseline;
  gap: var(--sp-5);
  margin-bottom: var(--sp-3);
  flex-wrap: wrap;
}
.status-word {
  font-size: var(--text-hero);
  font-weight: 600;
  letter-spacing: -0.04em;
  line-height: 1;
}
.status-word.s-training { color: var(--s-training); }
.status-word.s-paused   { color: var(--s-paused); }
.status-word.s-stopped  { color: var(--s-stopped); }
.status-word.s-error    { color: var(--s-error); }
.status-word.s-done     { color: var(--s-done); }
.status-word.s-idle     { color: var(--text-faint); }

.status-flag {
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--text-muted);
  font-family: var(--font-mono);
  letter-spacing: 0.01em;
  align-self: center;
  padding-top: 4px;
}
.status-line {
  font-size: var(--text-sm);
  color: var(--text-muted);
  max-width: 60ch;
  text-wrap: balance;
}
.heartbeat-label {
  font-size: var(--text-xs);
  color: var(--text-faint);
  margin-top: var(--sp-5);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.heartbeat-value {
  font-size: var(--text-sm);
  font-variant-numeric: tabular-nums;
  color: ${isStale ? "var(--s-error)" : "var(--text-muted)"};
  font-family: var(--font-mono);
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.progress-section {
  margin-bottom: var(--sp-6);
}
.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: var(--sp-2);
}
.progress-label {
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.progress-nums {
  font-size: var(--text-sm);
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  font-family: var(--font-mono);
}
.progress-nums strong { color: var(--text); font-weight: 500; }
.bar-track {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  height: 6px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: var(--radius-sm);
  width: ${pct}%;
  min-width: ${step > 0 ? "4px" : "0"};
  transition: width 800ms var(--ease-out);
}
.progress-footer {
  display: flex;
  justify-content: space-between;
  margin-top: var(--sp-2);
  font-size: var(--text-xs);
  color: var(--text-faint);
  font-variant-numeric: tabular-nums;
}
.pct-label {
  font-family: var(--font-mono);
  color: var(--text-muted);
  font-weight: 500;
}

/* ── Metrics strip ────────────────────────────────────────────────────────── */
.metrics-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: var(--sp-6);
}
.metric {
  background: var(--bg-raised);
  padding: var(--sp-4) var(--sp-5);
}
.metric-label {
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: var(--sp-2);
}
.metric-value {
  font-size: var(--text-2xl);
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
.metric-unit {
  font-size: var(--text-xs);
  color: var(--text-faint);
  margin-top: var(--sp-1);
}

/* ── Two-column section ───────────────────────────────────────────────────── */
.two-col {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: var(--sp-5);
  margin-bottom: var(--sp-5);
}
@media (max-width: 720px) {
  .two-col { grid-template-columns: 1fr; }
}

/* ── Section blocks ───────────────────────────────────────────────────────── */
.section {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: var(--sp-5);
}
.section-header {
  padding: var(--sp-3) var(--sp-5);
  border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
}
.section-title {
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.section-body {
  padding: var(--sp-4) var(--sp-5);
  background: var(--bg);
}
.section-body-wide {
  padding: 0;
  background: var(--bg);
}

/* ── Sparkline ────────────────────────────────────────────────────────────── */
.spark-meta {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: var(--sp-3);
}
.spark-current {
  font-size: var(--text-xl);
  font-weight: 500;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}
.spark-unit { font-size: var(--text-sm); color: var(--text-faint); font-weight: 400; }
.spark-trend {
  font-size: var(--text-sm);
  color: var(--text-faint);
  font-variant-numeric: tabular-nums;
}
.trend-good { color: var(--s-training); }
.trend-bad  { color: var(--s-error); }
svg { display: block; width: 100%; height: 56px; overflow: visible; }

/* ── Language table ───────────────────────────────────────────────────────── */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}
.data-table th {
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: var(--sp-3) var(--sp-5);
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
}
.data-table td {
  padding: var(--sp-3) var(--sp-5);
  border-bottom: 1px solid var(--border);
  color: var(--text-muted);
  vertical-align: middle;
}
.data-table tr:last-child td { border-bottom: none; }
.td-num {
  text-align: right;
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  font-variant-numeric: tabular-nums;
  color: var(--text);
}
.muted { color: var(--text-faint); }
.empty-cell { color: var(--text-faint); font-style: italic; padding: var(--sp-5); }

/* ── Alerts ───────────────────────────────────────────────────────────────── */
.alert-item {
  font-size: var(--text-xs);
  font-family: var(--font-mono);
  padding: var(--sp-2) var(--sp-5);
  border-bottom: 1px solid var(--border);
  color: var(--text-muted);
  line-height: 1.5;
}
.alert-item:last-child { border-bottom: none; }
.alert-warn { color: var(--s-paused); }
.alert-info { }

/* ── Controls ─────────────────────────────────────────────────────────────── */
.controls-body {
  padding: var(--sp-5);
  background: var(--bg);
}
.token-label {
  display: block;
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: var(--sp-2);
}
.token-input {
  width: 100%;
  height: 40px;
  padding: 0 var(--sp-3);
  background: var(--bg-raised);
  border: 1px solid var(--border-mid);
  border-radius: var(--radius-sm);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  outline: none;
  transition: border-color var(--dur-fast) var(--ease-out);
  margin-bottom: var(--sp-4);
}
.token-input:focus { border-color: var(--accent); }
.token-input::placeholder { color: var(--text-faint); }
.btn-row {
  display: flex;
  gap: var(--sp-2);
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--sp-2);
  height: 40px;
  padding: 0 var(--sp-4);
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  font-weight: 500;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition:
    background-color var(--dur-fast) var(--ease-out),
    border-color var(--dur-fast) var(--ease-out),
    color var(--dur-fast) var(--ease-out);
  letter-spacing: -0.01em;
}
.btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.btn:active { transform: translateY(1px); }
.btn[disabled] { opacity: 0.45; cursor: not-allowed; }
/* Primary — accent fill */
.btn-primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
.btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
/* Secondary — outlined */
.btn-secondary { background: var(--bg-raised); color: var(--text); border-color: var(--border-mid); }
.btn-secondary:hover { background: var(--bg-elevated); border-color: var(--border-mid); }
/* Danger — outlined in danger tone */
.btn-danger { background: var(--bg-raised); color: var(--s-error); border-color: rgba(255,255,255,0.08); }
.btn-danger:hover { background: oklch(20% 0.06 22); border-color: var(--s-error); }

.ctrl-msg {
  margin-top: var(--sp-3);
  font-size: var(--text-xs);
  font-family: var(--font-mono);
  min-height: 1.4em;
  color: var(--accent);
}

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
  font-size: var(--text-sm);
  color: var(--text-faint);
  font-style: italic;
  padding: var(--sp-2) 0;
}

/* ── Checkpoint hash ──────────────────────────────────────────────────────── */
.checkpoint-hash {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--text-faint);
  word-break: break-all;
  margin-top: var(--sp-2);
}

/* ── Full-width section override ──────────────────────────────────────────── */
.full { margin-bottom: var(--sp-5); }

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width: 600px) {
  .metrics-strip { grid-template-columns: 1fr 1fr; }
  .status-row { gap: var(--sp-3); }
  .topbar { padding-bottom: var(--sp-4); margin-bottom: var(--sp-4); }
}

@media (prefers-reduced-motion: reduce) {
  .bar-fill { transition: none; }
  .btn { transition: none; }
}
</style>
</head>
<body>
<div class="shell">

  <!-- Top bar -->
  <header class="topbar">
    <div class="brand">E-Brain 315M <span>~315M params · D=1024 L=24 H=16 T=4096 · 16K BPE · FlashAttention</span></div>
    <div class="topbar-right" id="refresh-label">Refreshing in <strong id="countdown">30</strong>s</div>
  </header>

  <!-- Status hero — the one focal point -->
  <section class="status-hero">
    <div class="status-row">
      <h1 class="status-word ${statusClass}">${status}</h1>
      <span class="status-flag">flag: ${flag}</span>
    </div>
    <p class="status-line">${statusLine}</p>
    <p class="heartbeat-label">Last heartbeat</p>
    <p class="heartbeat-value">${freshLabel}${isStale ? " — no signal for more than 15 min" : ""}</p>
  </section>

  <!-- Progress -->
  <section class="progress-section">
    <div class="progress-header">
      <span class="progress-label">Progress</span>
      <span class="progress-nums"><strong>${step.toLocaleString()}</strong> / ${TOTAL_STEPS.toLocaleString()} steps · 6.36B token target</span>
    </div>
    <div class="bar-track"><div class="bar-fill"></div></div>
    <div class="progress-footer">
      <span>${hb.eta_h !== undefined ? `~${Number(hb.eta_h).toFixed(1)}h remaining` : "ETA not available"}</span>
      <span class="pct-label">${pct}%</span>
    </div>
  </section>

  <!-- Metrics strip -->
  <div class="metrics-strip" role="list">
    <div class="metric" role="listitem">
      <div class="metric-label">Loss</div>
      <div class="metric-value">${fmtF(hb.loss, 4)}</div>
      <div class="metric-unit">cross-entropy</div>
    </div>
    <div class="metric" role="listitem">
      <div class="metric-label">Learning rate</div>
      <div class="metric-value" style="font-size: var(--text-lg)">${fmtE(hb.lr)}</div>
      <div class="metric-unit">cosine decay</div>
    </div>
    <div class="metric" role="listitem">
      <div class="metric-label">Throughput</div>
      <div class="metric-value">${fmtI(hb.tok_s)}</div>
      <div class="metric-unit">tokens / sec</div>
    </div>
    <div class="metric" role="listitem">
      <div class="metric-label">VRAM</div>
      <div class="metric-value">${fmtF(hb.vram_gb, 1)}</div>
      <div class="metric-unit">GB used</div>
    </div>
    <div class="metric" role="listitem">
      <div class="metric-label">Checkpoint</div>
      <div class="metric-value" style="font-size: var(--text-sm); font-family: var(--font-mono); font-weight: 400; color: var(--text-muted)">${hb.checkpoint_hash ? hb.checkpoint_hash.substring(0, 12) + "…" : "—"}</div>
      <div class="metric-unit">SHA-256 prefix</div>
    </div>
  </div>

  <!-- Loss curve + Language BPB -->
  <div class="two-col">

    <div class="section">
      <div class="section-header"><span class="section-title">Val loss curve — mean BPB, last ${sparkPts.length || 0} evals</span></div>
      <div class="section-body">${sparkSVG}</div>
    </div>

    <div class="section">
      <div class="section-header"><span class="section-title">Per-language BPB — latest eval</span></div>
      <div class="section-body-wide">
        <table class="data-table" aria-label="Per-language validation BPB">
          <thead><tr><th>Language</th><th style="text-align:right">BPB</th><th style="text-align:right">Change</th></tr></thead>
          <tbody>${langRows}</tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- Controls -->
  <div class="section full">
    <div class="section-header"><span class="section-title">Training controls</span></div>
    <div class="controls-body">
      <label class="token-label" for="dash_token">Dashboard token</label>
      <input type="password" class="token-input" id="dash_token" placeholder="Paste token to unlock controls" autocomplete="current-password">
      <div class="btn-row">
        <button class="btn btn-primary" onclick="setControl('RUN')">Resume training</button>
        <button class="btn btn-secondary" onclick="setControl('PAUSE')">Pause</button>
        <button class="btn btn-danger" onclick="confirmStop()">Stop and checkpoint</button>
      </div>
      <div class="ctrl-msg" id="ctrl_msg" role="status" aria-live="polite"></div>
    </div>
  </div>

  <!-- Alerts + Session history -->
  <div class="two-col">

    <div class="section">
      <div class="section-header"><span class="section-title">Recent alerts</span></div>
      <div class="section-body-wide">${alertRows}</div>
    </div>

    <div class="section">
      <div class="section-header"><span class="section-title">Session history</span></div>
      <div class="section-body-wide">
        <table class="data-table" aria-label="Training session history">
          <thead><tr><th>Date</th><th style="text-align:right">Steps</th><th style="text-align:right">Tokens</th><th style="text-align:right">Mean BPB</th></tr></thead>
          <tbody>${sessRows}</tbody>
        </table>
      </div>
    </div>

  </div>

</div><!-- /shell -->

<script>
// ── Controls ──────────────────────────────────────────────────────────────
async function setControl(flag) {
  const token = document.getElementById('dash_token').value.trim();
  if (!token) { showMsg('Paste your dashboard token first.', 'var(--s-paused)'); return; }
  showMsg('Sending…', 'var(--text-faint)');
  try {
    const r = await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Token': token },
      body: JSON.stringify({ control: flag })
    });
    const d = await r.json();
    if (d.ok) showMsg('Set to ' + flag + '. Page will refresh.', 'var(--accent)');
    else showMsg('Error: ' + (d.error || r.status), 'var(--s-error)');
  } catch (e) { showMsg('Network error.', 'var(--s-error)'); }
}

function confirmStop() {
  if (confirm('Stop training?\n\nThe kernel will save a checkpoint then exit cleanly. You can resume from that checkpoint later.')) {
    setControl('STOP');
  }
}

function showMsg(msg, color) {
  const el = document.getElementById('ctrl_msg');
  el.textContent = msg;
  el.style.color = color;
}

// ── Countdown refresh ─────────────────────────────────────────────────────
let s = 30;
(function tick() {
  s--;
  const el = document.getElementById('countdown');
  if (el) el.textContent = s;
  if (s <= 0) { location.reload(); return; }
  setTimeout(tick, 1000);
})();
</script>
</body>
</html>`);
}

// ─── Router ───────────────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const url      = new URL(request.url);
    const { pathname } = url;
    const method   = request.method.toUpperCase();

    if (method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin":  "*",
          "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type,X-Token",
        },
      });
    }

    if (pathname === "/heartbeat"     && method === "POST") return handleHeartbeat(request, env);
    if (pathname === "/status"        && method === "GET")  return handleGetStatus(env);
    if (pathname === "/control"       && method === "GET")  return handleGetControl(env);
    if (pathname === "/control"       && method === "POST") return handlePostControl(request, env);
    if (pathname === "/telegram_test" && method === "POST") return handleTelegramTest(request, env);
    if (pathname === "/"              && method === "GET")  return handleDashboard(env);

    return new Response("Not Found", { status: 404 });
  },
};
