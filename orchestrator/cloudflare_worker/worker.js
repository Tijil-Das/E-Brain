/**
 * E-Brain 100M Training Orchestrator — Cloudflare Worker
 *
 * Endpoints:
 *   POST /heartbeat      (auth: HEARTBEAT_TOKEN header)
 *   GET  /status         (public JSON)
 *   GET  /control        (public JSON  — polled by training kernel)
 *   POST /control        (auth: DASHBOARD_TOKEN header)
 *   GET  /               (dashboard HTML, password-protected)
 *   POST /telegram_test  (auth: DASHBOARD_TOKEN — sends a test message)
 *
 * KV namespace: EBRAIN_KV  (binding name in wrangler.toml)
 * Secrets (set via `wrangler secret put`):
 *   HEARTBEAT_TOKEN    — shared secret kernel sends in X-Token header
 *   DASHBOARD_TOKEN    — dashboard POST auth token
 *   TELEGRAM_BOT_TOKEN — optional; Telegram bot token
 *   TELEGRAM_CHAT_ID   — optional; Telegram chat ID
 *
 * KV keys:
 *   control_flag          "RUN" | "PAUSE" | "STOP"
 *   last_heartbeat        JSON HeartbeatPayload
 *   val_history           JSON array of ValEntry
 *   session_history       JSON array of SessionEntry
 *   alert_log             JSON array of strings (last 50)
 */

const TOTAL_STEPS = 73242;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
}

function html(body, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}

function requireHeader(request, env, headerName, envVar) {
  const token = request.headers.get(headerName);
  if (!token || token !== env[envVar]) {
    return new Response("Unauthorized", { status: 401 });
  }
  return null; // ok
}

async function sendTelegram(env, message) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID) return;
  try {
    await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: env.TELEGRAM_CHAT_ID,
          text: `🤖 E-Brain 100M\n${message}`,
          parse_mode: "HTML",
        }),
      }
    );
  } catch (_) {}
}

async function appendAlert(env, message) {
  let log = [];
  try { log = JSON.parse(await env.EBRAIN_KV.get("alert_log") || "[]"); } catch (_) {}
  log.unshift(`[${new Date().toISOString()}] ${message}`);
  if (log.length > 50) log = log.slice(0, 50);
  await env.EBRAIN_KV.put("alert_log", JSON.stringify(log));
}

// ─── Handlers ────────────────────────────────────────────────────────────────

async function handleHeartbeat(request, env) {
  const authErr = requireHeader(request, env, "X-Token", "HEARTBEAT_TOKEN");
  if (authErr) return authErr;

  let payload;
  try { payload = await request.json(); }
  catch (_) { return json({ error: "invalid JSON" }, 400); }

  // Validate required fields
  const required = ["step", "loss", "lr", "tok_s", "vram_gb", "eta_h"];
  for (const f of required) {
    if (payload[f] === undefined) return json({ error: `missing field: ${f}` }, 400);
  }

  payload.received_at = new Date().toISOString();
  await env.EBRAIN_KV.put("last_heartbeat", JSON.stringify(payload));

  // Alert checks
  const alerts = [];
  if (payload.nan_alert) {
    alerts.push(`⚠️ NaN detected at step ${payload.step}`);
  }
  if (payload.loss_spike) {
    alerts.push(`⚠️ Loss spike at step ${payload.step}: loss=${payload.loss.toFixed(4)}`);
  }
  if (payload.val_alert) {
    alerts.push(`⚠️ Val degraded at step ${payload.step}: ${payload.val_alert}`);
  }
  if (payload.lang_gap_alert) {
    alerts.push(`⚠️ Language gap >0.15 at step ${payload.step}: ${payload.lang_gap_alert}`);
  }

  for (const a of alerts) {
    await appendAlert(env, a);
    await sendTelegram(env, a);
  }

  // Append val entry if present
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
  const hb = JSON.parse(await env.EBRAIN_KV.get("last_heartbeat") || "{}");
  const flag = await env.EBRAIN_KV.get("control_flag") || "RUN";
  const valHist = JSON.parse(await env.EBRAIN_KV.get("val_history") || "[]");
  const sessions = JSON.parse(await env.EBRAIN_KV.get("session_history") || "[]");
  const alerts = JSON.parse(await env.EBRAIN_KV.get("alert_log") || "[]");

  const step = hb.step || 0;
  const pct = ((step / TOTAL_STEPS) * 100).toFixed(2);

  return json({
    control_flag: flag,
    step,
    total_steps: TOTAL_STEPS,
    progress_pct: parseFloat(pct),
    loss: hb.loss,
    lr: hb.lr,
    tok_s: hb.tok_s,
    vram_gb: hb.vram_gb,
    eta_h: hb.eta_h,
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
  try { body = await request.json(); } catch (_) { return json({ error: "invalid JSON" }, 400); }

  const { control } = body;
  if (!["RUN", "PAUSE", "STOP"].includes(control)) {
    return json({ error: "control must be RUN | PAUSE | STOP" }, 400);
  }

  await env.EBRAIN_KV.put("control_flag", control);
  const msg = `🎛️ Control set to ${control} by dashboard`;
  await appendAlert(env, msg);
  await sendTelegram(env, msg);

  return json({ ok: true, control });
}

async function handleTelegramTest(request, env) {
  const authErr = requireHeader(request, env, "X-Token", "DASHBOARD_TOKEN");
  if (authErr) return authErr;
  await sendTelegram(env, "✅ Telegram test message from E-Brain dashboard");
  return json({ ok: true });
}

// ─── Dashboard HTML ───────────────────────────────────────────────────────────

async function handleDashboard(env) {
  const hb = JSON.parse(await env.EBRAIN_KV.get("last_heartbeat") || "{}");
  const flag = await env.EBRAIN_KV.get("control_flag") || "RUN";
  const valHist = JSON.parse(await env.EBRAIN_KV.get("val_history") || "[]");
  const alerts = JSON.parse(await env.EBRAIN_KV.get("alert_log") || "[]");
  const sessions = JSON.parse(await env.EBRAIN_KV.get("session_history") || "[]");

  const step = hb.step || 0;
  const pct = ((step / TOTAL_STEPS) * 100).toFixed(2);
  const lastHB = hb.received_at ? new Date(hb.received_at).toLocaleString() : "—";

  // Staleness check
  let hbStale = false;
  if (hb.received_at) {
    hbStale = (Date.now() - new Date(hb.received_at).getTime()) > 15 * 60 * 1000;
  }

  // Per-language val table
  const LANGS = ["en", "es", "ru", "fr", "de", "ja", "zh", "hi", "bn", "ar"];
  let valRows = "";
  if (hb.val_per_lang) {
    for (const lang of LANGS) {
      const bpb = hb.val_per_lang[lang];
      if (bpb !== undefined) {
        // Find trend: compare to previous val entry
        let trend = "—";
        if (valHist.length >= 2) {
          const prev = valHist[valHist.length - 2]?.bpb?.[lang];
          if (prev !== undefined) {
            trend = bpb < prev ? `▼ ${(prev - bpb).toFixed(4)}` : `▲ ${(bpb - prev).toFixed(4)}`;
          }
        }
        valRows += `<tr><td>${lang}</td><td>${bpb.toFixed(4)}</td><td>${trend}</td></tr>`;
      }
    }
  } else {
    valRows = `<tr><td colspan="3" style="color:#888">No val data yet</td></tr>`;
  }

  // Loss sparkline from val history (last 20 mean bpb values)
  const sparkPoints = valHist.slice(-20).map(v => {
    if (!v.bpb) return null;
    const vals = Object.values(v.bpb);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }).filter(x => x !== null);

  let sparkSVG = "";
  if (sparkPoints.length >= 2) {
    const minV = Math.min(...sparkPoints);
    const maxV = Math.max(...sparkPoints) || minV + 0.01;
    const W = 300, H = 60, pad = 4;
    const pts = sparkPoints.map((v, i) => {
      const x = pad + (i / (sparkPoints.length - 1)) * (W - 2 * pad);
      const y = H - pad - ((v - minV) / (maxV - minV)) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    sparkSVG = `<svg width="${W}" height="${H}" style="background:#1a1a2e;border-radius:6px">
      <polyline points="${pts}" fill="none" stroke="#4ecca3" stroke-width="2"/>
    </svg>`;
  }

  // Alert log
  const alertHTML = alerts.slice(0, 10).map(a =>
    `<div class="alert">${a}</div>`).join("") || "<div style='color:#888'>No alerts</div>";

  // Session history
  const sessHTML = sessions.slice(-10).reverse().map(s =>
    `<tr><td>${s.date || "—"}</td><td>${s.steps || 0}</td><td>${s.tokens_B?.toFixed(3) || "—"}B</td><td>${s.mean_val_bpb?.toFixed(4) || "—"}</td></tr>`
  ).join("") || `<tr><td colspan="4" style="color:#888">No sessions yet</td></tr>`;

  const flagColor = { RUN: "#4ecca3", PAUSE: "#f5a623", STOP: "#e74c3c" }[flag] || "#888";

  return html(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E-Brain 100M Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0d1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}
  h1{color:#4ecca3;font-size:1.6rem;margin-bottom:4px}
  .sub{color:#888;font-size:.85rem;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
  .card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px}
  .card h3{color:#888;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
  .card .val{font-size:1.5rem;font-weight:700;color:#fff}
  .card .sub2{font-size:.8rem;color:#888;margin-top:3px}
  .progress-bar{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px;margin-bottom:20px}
  .bar-track{background:#0d0d1a;border-radius:50px;height:20px;overflow:hidden;margin-top:8px}
  .bar-fill{background:linear-gradient(90deg,#4ecca3,#1a7a5e);height:100%;border-radius:50px;transition:width .5s}
  .bar-label{text-align:right;font-size:.8rem;color:#4ecca3;margin-top:4px}
  .section{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px;margin-bottom:16px}
  .section h2{font-size:1rem;color:#4ecca3;margin-bottom:12px}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{color:#888;text-align:left;padding:4px 8px;border-bottom:1px solid #2a2a4a}
  td{padding:4px 8px;border-bottom:1px solid #111}
  .flag{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;color:#000;font-size:.9rem}
  .btn{padding:8px 18px;border:none;border-radius:6px;font-size:.9rem;font-weight:600;cursor:pointer;margin-right:8px}
  .btn-run{background:#4ecca3;color:#000}
  .btn-pause{background:#f5a623;color:#000}
  .btn-stop{background:#e74c3c;color:#fff}
  .btn:hover{opacity:.85}
  .stale{color:#e74c3c;font-size:.8rem}
  .alert{font-size:.8rem;color:#f5a623;padding:3px 0;border-bottom:1px solid #111}
  input[type=password]{background:#0d0d1a;border:1px solid #2a2a4a;color:#fff;padding:6px 10px;border-radius:6px;width:220px}
  .tok-acct{margin-top:14px}
  .tok-acct a{color:#4ecca3;text-decoration:none}
</style>
</head>
<body>
<h1>🧠 E-Brain 100M Training Dashboard</h1>
<div class="sub">97.24M params · D=768 L=12 H=12 T=2048 · 16K BPE · 2.40B tokens target · 73,242 steps</div>

<div class="grid">
  <div class="card">
    <h3>Control Flag</h3>
    <div class="val"><span class="flag" style="background:${flagColor}">${flag}</span></div>
  </div>
  <div class="card">
    <h3>Step</h3>
    <div class="val">${step.toLocaleString()}</div>
    <div class="sub2">of ${TOTAL_STEPS.toLocaleString()}</div>
  </div>
  <div class="card">
    <h3>Loss</h3>
    <div class="val">${hb.loss !== undefined ? hb.loss.toFixed(4) : "—"}</div>
  </div>
  <div class="card">
    <h3>Throughput</h3>
    <div class="val">${hb.tok_s !== undefined ? Math.round(hb.tok_s).toLocaleString() : "—"}</div>
    <div class="sub2">tok/s</div>
  </div>
  <div class="card">
    <h3>VRAM</h3>
    <div class="val">${hb.vram_gb !== undefined ? hb.vram_gb.toFixed(2) : "—"}</div>
    <div class="sub2">GB</div>
  </div>
  <div class="card">
    <h3>ETA</h3>
    <div class="val">${hb.eta_h !== undefined ? hb.eta_h.toFixed(1) : "—"}</div>
    <div class="sub2">hours remaining</div>
  </div>
  <div class="card">
    <h3>Last Heartbeat</h3>
    <div class="val" style="font-size:.95rem">${lastHB}</div>
    ${hbStale ? '<div class="stale">⚠️ No heartbeat for >15 min</div>' : ""}
  </div>
  <div class="card">
    <h3>Checkpoint</h3>
    <div class="val" style="font-size:.7rem;word-break:break-all">${hb.checkpoint_hash ? hb.checkpoint_hash.substring(0, 16) + "..." : "—"}</div>
  </div>
</div>

<div class="progress-bar">
  <strong>Training Progress</strong>
  <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
  <div class="bar-label">${pct}% · ${step.toLocaleString()} / ${TOTAL_STEPS.toLocaleString()} steps</div>
</div>

<div class="section">
  <h2>📉 Loss Curve (mean val BPB, last 20 evals)</h2>
  ${sparkSVG || '<div style="color:#888">No data yet</div>'}
</div>

<div class="section">
  <h2>🌍 Per-Language Val BPB (latest eval)</h2>
  <table>
    <tr><th>Lang</th><th>BPB</th><th>Trend</th></tr>
    ${valRows}
  </table>
</div>

<div class="section">
  <h2>📋 Session History</h2>
  <table>
    <tr><th>Date</th><th>Steps</th><th>Tokens</th><th>Mean Val BPB</th></tr>
    ${sessHTML}
  </table>
</div>

<div class="section">
  <h2>🚨 Recent Alerts</h2>
  ${alertHTML}
</div>

<div class="section">
  <h2>🎛️ Controls</h2>
  <p style="font-size:.8rem;color:#888;margin-bottom:10px">Enter your dashboard token to send control commands.</p>
  <input type="password" id="dash_token" placeholder="Dashboard token">
  <div style="margin-top:12px">
    <button class="btn btn-run" onclick="setControl('RUN')">▶ Resume</button>
    <button class="btn btn-pause" onclick="setControl('PAUSE')">⏸ Pause</button>
    <button class="btn btn-stop" onclick="confirmStop()">⏹ Stop</button>
  </div>
  <div id="ctrl_msg" style="margin-top:8px;font-size:.85rem;color:#4ecca3"></div>

  <div class="tok-acct" style="margin-top:20px">
    <strong>Kaggle Training Account:</strong>
    <a href="https://www.kaggle.com" target="_blank">Open Kaggle →</a>
    <span style="color:#888;font-size:.8rem;margin-left:8px">(log in with the dedicated training account)</span>
  </div>
</div>

<script>
async function setControl(flag) {
  const token = document.getElementById('dash_token').value;
  if (!token) { showMsg('Enter your dashboard token first', 'orange'); return; }
  try {
    const r = await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Token': token },
      body: JSON.stringify({ control: flag })
    });
    const d = await r.json();
    if (d.ok) showMsg('✅ Control set to ' + flag, '#4ecca3');
    else showMsg('Error: ' + (d.error || r.status), 'red');
  } catch(e) { showMsg('Network error: ' + e, 'red'); }
}

function confirmStop() {
  if (confirm('⚠️ STOP training? The kernel will checkpoint and exit. This cannot be undone without manual Resume.')) {
    setControl('STOP');
  }
}

function showMsg(msg, color) {
  const el = document.getElementById('ctrl_msg');
  el.textContent = msg;
  el.style.color = color || '#4ecca3';
}

// Auto-refresh every 60s
setTimeout(() => location.reload(), 60000);
</script>
</body>
</html>`);
}

// ─── Router ───────────────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;
    const method = request.method.toUpperCase();

    if (method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type,X-Token",
        },
      });
    }

    if (pathname === "/heartbeat" && method === "POST") return handleHeartbeat(request, env);
    if (pathname === "/status"    && method === "GET")  return handleGetStatus(env);
    if (pathname === "/control"   && method === "GET")  return handleGetControl(env);
    if (pathname === "/control"   && method === "POST") return handlePostControl(request, env);
    if (pathname === "/telegram_test" && method === "POST") return handleTelegramTest(request, env);
    if (pathname === "/"          && method === "GET")  return handleDashboard(env);

    return new Response("Not Found", { status: 404 });
  },
};
