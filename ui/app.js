/* Fraud Intel UI: feed, graph, cases, detail, replay controls. No build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { cursor: 0, total: 0, network: null, showAll: false };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtAmt(e) {
  return e.amount == null ? "?" : `Rs ${Number(e.amount).toLocaleString("en-IN")}`;
}

function bandOf(e) { return String(e.txn_risk_band || e.severity || "low").toLowerCase(); }

function pill(band) {
  const b = bandOf({ txn_risk_band: band });
  return `<span class="pill ${b}">${esc(band)}</span>`;
}

function scoreBar(score, band) {
  const colors = { critical: "#f44", high: "#f80", medium: "#dd4", low: "#4a4" };
  const v = Math.max(0, Math.min(100, Number(score) || 0));
  return `<span class="scorebar"><span style="width:${v}%;background:${colors[bandOf({ txn_risk_band: band })] || "#888"}"></span></span>`;
}

async function refreshStatus() {
  try {
    const h = await api("/v1/health");
    state.cursor = h.cursor; state.total = h.n_events;
    $("status-text").textContent =
      `${h.playing ? "LIVE" : "paused"} · cursor ${h.cursor}/${h.n_events} · speed ${h.speed}×`;
    $("status").classList.toggle("live", !!h.playing);
  } catch (e) {
    $("status-text").textContent = "server unreachable";
    $("status").classList.remove("live");
  }
}

async function control(op) {
  const speed = Number($("sel-speed").value);
  await api("/v1/replay/control", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op, speed }),
  });
  await refreshAll();
}

function txnRow(e) {
  const band = bandOf(e);
  const live = e.live ? `<span class="chip live">LIVE</span>` : "";
  return `<tr data-txn="${esc(e.txn_id)}">` +
    `<td class="mono">${esc((e.timestamp || "").slice(0, 16).replace("T", " "))}</td>` +
    `<td class="mono">${esc(e.txn_id)} ${live}</td><td class="mono">${esc(e.account_id)}</td>` +
    `<td>${esc(fmtAmt(e))}</td>` +
    `<td class="mono">${esc(e.txn_risk_score)}${scoreBar(e.txn_risk_score, band)}</td>` +
    `<td>${pill(e.txn_risk_band || "?")}</td></tr>`;
}

async function refreshFeed() {
  const from = Math.max(0, state.total - 50);
  const { events } = await api(`/v1/events?cursor=${from}&limit=50`);
  $("feed-count").textContent = events.length;
  $("feed").innerHTML =
    "<thead><tr><th>time</th><th>txn</th><th>account</th><th>amount</th><th>score</th><th>band</th></tr></thead><tbody>" +
    (events.map(txnRow).join("") || `<tr><td colspan="6" class="empty">no events yet — press Start</td></tr>`) +
    "</tbody>";
  $("feed").querySelectorAll("tr[data-txn]").forEach((tr) => {
    tr.onclick = () => showEntity(tr.dataset.txn);
  });
  return events;
}

function isTransfer(e) {
  return (e.txn_type === "transfer" || e.txn_type === "cash_out") && e.dest_account_id;
}

const NODE_COLORS = { critical: "#f44", high: "#f80", medium: "#dd4", low: "#4a4" };

function refreshGraph(events) {
  const sel = events.filter((e) => state.showAll || isTransfer(e)).slice(-40);
  $("graph-count").textContent = sel.length;
  const nodes = new Map(), edges = [];
  for (const e of sel) {
    const a = String(e.account_id);
    if (!nodes.has(a)) nodes.set(a, {
      id: a, label: a.length > 10 ? a.slice(-8) : a,
      color: { background: "#1a2130", border: NODE_COLORS[bandOf(e)] || "#888" },
      font: { color: "#dbe2ef" }, shape: "dot", size: 14,
      title: `${a} (risk ${e.txn_risk_score})`,
    });
    const d = e.dest_account_id ? String(e.dest_account_id) : null;
    if (d) {
      if (!nodes.has(d)) nodes.set(d, {
        id: d, label: d.length > 10 ? d.slice(-8) : d,
        color: { background: "#12283a", border: "#48c" },
        font: { color: "#dbe2ef" }, shape: "diamond", size: 14, title: d,
      });
      edges.push({ from: a, to: d, color: { color: "#3a4a6b" },
        title: `${e.txn_id}: ${fmtAmt(e)} (risk ${e.txn_risk_score})` });
    }
  }
  const data = { nodes: new vis.DataSet([...nodes.values()]), edges: new vis.DataSet(edges) };
  const options = {
    physics: { barnesHut: { gravitationalConstant: -4000 }, stabilization: true },
    interaction: { hover: true, tooltipDelay: 100 },
  };
  if (!state.network) {
    state.network = new vis.Network($("graph"), data, options);
    state.network.on("click", (p) => { if (p.nodes.length) showEntity(p.nodes[0]); });
  } else {
    state.network.setData(data);
  }
  if (!sel.length) $("graph-count").textContent = "0 — no transfers in window";
}

async function refreshCases() {
  const { cases } = await api("/v1/cases");
  $("case-count").textContent = cases.length;
  $("cases").innerHTML = cases.map((c) => {
    const sev = String(c.severity || "?");
    const nEnt = (c.entities || []).length, nTxn = (c.transaction_ids || []).length;
    const typs = (c.typologies || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("");
    return `<div class="case" data-case="${esc(c.case_id)}">` +
      `<div class="row1"><span class="cid">${esc(c.case_id)}</span>${pill(sev)}</div>` +
      `<div class="meta">${nEnt} entities · ${nTxn} txns<br>${typs}</div></div>`;
  }).join("") || `<div class="empty">no cases</div>`;
  $("cases").querySelectorAll(".case").forEach((el) => {
    el.onclick = async () => {
      const c = await api(`/v1/cases/${encodeURIComponent(el.dataset.case)}`);
      renderDetail({ kind: "case", ...c });
    };
  });
}

function kvRows(obj, keys) {
  return keys.filter((k) => obj[k] != null && obj[k] !== "")
    .map((k) => `<dt>${esc(k)}</dt><dd>${esc(obj[k])}</dd>`).join("");
}

function renderDetail(ent) {
  const parts = [];
  if (ent.kind === "account") {
    const band = String(ent.account_risk_band || "?");
    parts.push(`<div class="dhead"><span class="did">${esc(ent.account_id)}</span>` +
      `${pill(band)}<span class="mono">score ${esc(ent.account_risk_score)}</span>` +
      (ent.ring_id ? `<span class="chip">ring: ${esc(ent.ring_id)}</span>` : "") +
      `<span class="chip">action: ${esc(ent.account_action || "")}</span></div>`);
    if (ent.account_reasons) {
      parts.push("<ul class='reasons'>" +
        String(ent.account_reasons).split(";").map((s) => `<li>${esc(s.trim())}</li>`).join("") + "</ul>");
    }
    parts.push(`<dl>${kvRows(ent, ["n_txns", "segment", "home_city"])}</dl>`);
    if (ent.transactions && ent.transactions.length) {
      parts.push(`<div class="meta" style="color:var(--muted)">${ent.transactions.length} transactions ` +
        `(latest: <span class="mono">${esc(ent.transactions.slice(-3).join(", "))}</span>)</div>`);
    }
  } else if (ent.kind === "transaction") {
    const band = bandOf(ent);
    parts.push(`<div class="dhead"><span class="did">${esc(ent.txn_id)}</span>` +
      `${pill(ent.txn_risk_band || "?")}<span class="mono">score ${esc(ent.txn_risk_score)}</span>` +
      `<span class="chip">${esc(fmtAmt(ent))}</span>` +
      `<span class="chip">action: ${esc(ent.txn_action || "")}</span></div>`);
    if (ent.txn_reasons) {
      parts.push("<ul class='reasons'>" +
        String(ent.txn_reasons).split(";").map((s) => `<li>${esc(s.trim())}</li>`).join("") + "</ul>");
    }
    parts.push(`<dl>${kvRows(ent, ["account_id", "timestamp", "txn_type", "channel",
      "merchant_category", "city", "device_id", "ip_address", "dest_account_id"])}</dl>`);
  } else {
    parts.push(`<div class="dhead"><span class="did">${esc(ent.case_id || "?")}</span>` +
      `${pill(ent.severity || "?")}</div>`);
    if (ent.severity_reason) parts.push(`<div>${esc(ent.severity_reason)}</div>`);
  }
  parts.push(`<details><summary>raw JSON</summary><pre>${esc(JSON.stringify(ent, null, 1))}</pre></details>`);
  $("detail").innerHTML = parts.join("");
}

async function showEntity(id) {
  try {
    renderDetail(await api(`/v1/entities/${encodeURIComponent(id)}`));
  } catch (e) { $("detail").innerHTML = `<div class="empty">entity not found: ${esc(id)}</div>`; }
}

async function refreshAll() {
  await refreshStatus();
  const events = await refreshFeed();
  refreshGraph(events);
  await refreshCases();
}

$("btn-start").onclick = () => control("start");
$("btn-pause").onclick = () => control("pause");
$("btn-reset").onclick = () => control("reset");
$("chk-all").onchange = (e) => { state.showAll = e.target.checked; refreshAll(); };

// Live updates via SSE (falls back to polling every 5s on error).
function connectStream() {
  try {
    const src = new EventSource("/v1/events/stream");
    src.onmessage = () => refreshAll();
    src.onerror = () => { src.close(); setTimeout(connectStream, 5000); };
  } catch (e) { setInterval(refreshAll, 5000); }
}

refreshAll();
connectStream();
setInterval(refreshStatus, 5000);
