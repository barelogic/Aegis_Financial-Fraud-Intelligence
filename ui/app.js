/* Fraud Intel UI: feed, graph, cases, detail, replay controls. No build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { cursor: 0, total: 0, network: null, showAll: false };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function fmtAmt(e) {
  return e.amount == null ? "?" : `Rs ${Number(e.amount).toLocaleString("en-IN")}`;
}

async function refreshStatus() {
  try {
    const h = await api("/v1/health");
    state.cursor = h.cursor; state.total = h.n_events;
    $("status").textContent =
      `${h.playing ? "LIVE" : "paused"}  cursor ${h.cursor}/${h.n_events}  speed ${h.speed}`;
  } catch (e) { $("status").textContent = "server unreachable"; }
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
  const band = (e.txn_risk_band || "low").toLowerCase();
  return `<tr class="${band}" data-txn="${e.txn_id}">` +
    `<td>${(e.timestamp || "").slice(0, 16).replace("T", " ")}</td>` +
    `<td>${e.txn_id}</td><td>${e.account_id}</td>` +
    `<td>${fmtAmt(e)}</td><td>${e.txn_risk_score}</td>` +
    `<td>${e.txn_action || ""}</td></tr>`;
}

async function refreshFeed() {
  const from = Math.max(0, state.total - 50);
  const { events } = await api(`/v1/events?cursor=${from}&limit=50`);
  $("feed").innerHTML =
    "<tr><th>time</th><th>txn</th><th>acct</th><th>amt</th><th>score</th><th>action</th></tr>" +
    events.map(txnRow).join("");
  $("feed").querySelectorAll("tr[data-txn]").forEach((tr) => {
    tr.onclick = () => showEntity(tr.dataset.txn);
  });
  return events;
}

function isTransfer(e) {
  return (e.txn_type === "transfer" || e.txn_type === "cash_out") && e.dest_account_id;
}

function refreshGraph(events) {
  const sel = events.filter((e) => state.showAll || isTransfer(e)).slice(-40);
  const nodes = new Map(), edges = [];
  const bandColor = { critical: "#f44", high: "#f80", medium: "#dd4", low: "#4a4" };
  for (const e of sel) {
    const a = String(e.account_id);
    if (!nodes.has(a)) nodes.set(a, { id: a, label: a.slice(-8), color: bandColor[e.txn_risk_band] || "#888" });
    const d = e.dest_account_id ? String(e.dest_account_id) : null;
    if (d) {
      if (!nodes.has(d)) nodes.set(d, { id: d, label: d.slice(-8), color: "#48c" });
      edges.push({ from: a, to: d, title: `${e.txn_id}: ${fmtAmt(e)} (risk ${e.txn_risk_score})` });
    }
  }
  const data = { nodes: new vis.DataSet([...nodes.values()]), edges: new vis.DataSet(edges) };
  if (!state.network) {
    state.network = new vis.Network($("graph"), data, { physics: { stabilization: true } });
    state.network.on("click", (p) => { if (p.nodes.length) showEntity(p.nodes[0]); });
  } else {
    state.network.setData(data);
  }
}

async function refreshCases() {
  const { cases } = await api("/v1/cases");
  $("cases").innerHTML = cases.map((c) =>
    `<div class="case" data-case="${c.case_id}"><b>${c.case_id}</b> [${c.severity}] ` +
    `${(c.entities || []).length} entities — ${(c.typologies || []).join(", ")}</div>`
  ).join("") || "<i>no cases</i>";
  $("cases").querySelectorAll(".case").forEach((el) => {
    el.onclick = async () => {
      const c = await api(`/v1/cases/${el.dataset.case}`);
      $("detail").textContent = JSON.stringify(c, null, 1);
    };
  });
}

async function showEntity(id) {
  try {
    const ent = await api(`/v1/entities/${encodeURIComponent(id)}`);
    $("detail").textContent = JSON.stringify(ent, null, 1);
  } catch (e) { $("detail").textContent = `entity not found: ${id}`; }
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
