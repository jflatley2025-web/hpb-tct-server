"""Index HTML template — self-contained browsable replay package listing."""
from __future__ import annotations

import html
import json

from local_chart_overlay.replay_index.models import (
    ReplayPackageEntry, ReplayIndexSummary,
)


def render_index_html(
    entries: list[ReplayPackageEntry],
    summary: ReplayIndexSummary,
    generated_at: str = "",
    input_dir_label: str = "",
) -> str:
    """Render a self-contained index.html for replay share packages.

    Features:
      - Summary metrics header
      - Text search
      - Filter by symbol, timeframe, side, flags
      - Sort by any column
      - Relative links to replay.html and open_chart.html
    """
    # Serialize entries to JSON for JS consumption
    entries_json = json.dumps([_entry_to_dict(e) for e in entries], default=str)
    entries_json_safe = entries_json.replace("</", "<\\/")

    symbols_json = json.dumps(summary.unique_symbols)
    timeframes_json = json.dumps(summary.unique_timeframes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Replay Package Index</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f1a;color:#d0d0e0;padding:24px}}
.header{{max-width:1200px;margin:0 auto 24px}}
.header h1{{font-size:22px;color:#fff;margin-bottom:4px}}
.header .meta{{font-size:12px;color:#6666aa;margin-bottom:16px}}
.summary{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}}
.stat{{background:#16162a;border:1px solid #2a2a4a;border-radius:8px;padding:12px 18px;min-width:120px}}
.stat .val{{font-size:22px;font-weight:700;color:#fff}}
.stat .lbl{{font-size:11px;color:#7777aa;margin-top:2px}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px;max-width:1200px;margin-left:auto;margin-right:auto}}
.search{{padding:8px 14px;border:1px solid #3a3a5a;border-radius:6px;background:#1a1a3a;color:#d0d0e0;font-size:13px;width:220px}}
.search:focus{{outline:none;border-color:#2962ff}}
select{{padding:7px 10px;border:1px solid #3a3a5a;border-radius:6px;background:#1a1a3a;color:#d0d0e0;font-size:12px;cursor:pointer}}
.sort-btn{{padding:6px 12px;border:1px solid #3a3a5a;border-radius:6px;background:#1a1a3a;color:#a0a0c0;cursor:pointer;font-size:12px}}
.sort-btn:hover,.sort-btn.active{{background:#2a2a5a;color:#fff}}
.table-wrap{{max-width:1200px;margin:0 auto;overflow-x:auto}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:8px 12px;font-size:11px;color:#7777aa;text-transform:uppercase;border-bottom:2px solid #2a2a4a;cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{color:#c0c0e0}}
td{{padding:10px 12px;border-bottom:1px solid #1a1a3a;font-size:13px;vertical-align:middle}}
tr:hover{{background:#16162a}}
.pill{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-right:3px}}
.pill-green{{background:#1b5e2040;color:#66bb6a;border:1px solid #2e7d3260}}
.pill-red{{background:#b7141440;color:#ef5350;border:1px solid #c6282860}}
.pill-blue{{background:#1565c040;color:#64b5f6;border:1px solid #1976d260}}
.pill-yellow{{background:#f9a82540;color:#ffd54f;border:1px solid #ffa00060}}
.pill-gray{{background:#44446640;color:#9999bb;border:1px solid #55558860}}
.link-btn{{padding:4px 10px;border:1px solid #3a3a5a;border-radius:5px;text-decoration:none;font-size:11px;font-weight:600;margin-right:4px;display:inline-block}}
.link-btn.primary{{background:#2962ff;border-color:#2962ff;color:#fff}}
.link-btn.primary:hover{{background:#1e4fd6}}
.link-btn.secondary{{background:#1a1a3a;color:#a0a0c0}}
.link-btn.secondary:hover{{background:#2a2a5a;color:#fff}}
.link-btn.disabled{{opacity:.3;pointer-events:none}}
.badge-row{{display:flex;gap:3px;flex-wrap:wrap}}
.empty{{text-align:center;padding:40px;color:#555}}
.footer{{max-width:1200px;margin:24px auto 0;font-size:11px;color:#444466;text-align:center}}
</style>
</head>
<body>

<div class="header">
  <h1>Replay Package Index</h1>
  <div class="meta">{html.escape(input_dir_label)} | Generated {html.escape(generated_at)} | {summary.total_packages} packages</div>
  <div class="summary">
    <div class="stat"><div class="val">{summary.total_packages}</div><div class="lbl">Packages</div></div>
    <div class="stat"><div class="val">{summary.total_trades}</div><div class="lbl">Trades</div></div>
    <div class="stat"><div class="val">{len(summary.unique_symbols)}</div><div class="lbl">Symbols</div></div>
    <div class="stat"><div class="val">{len(summary.unique_timeframes)}</div><div class="lbl">Timeframes</div></div>
    <div class="stat"><div class="val">{summary.with_confirmed}</div><div class="lbl">Confirmed</div></div>
    <div class="stat"><div class="val">{summary.with_suggested}</div><div class="lbl">Suggested</div></div>
    <div class="stat"><div class="val">{summary.with_accuracy}</div><div class="lbl">Accuracy</div></div>
  </div>
</div>

<div class="controls">
  <input class="search" type="text" id="searchBox" placeholder="Search..." oninput="applyFilters()">
  <select id="filterSymbol" onchange="applyFilters()"><option value="">All Symbols</option></select>
  <select id="filterTf" onchange="applyFilters()"><option value="">All Timeframes</option></select>
  <select id="filterSide" onchange="applyFilters()">
    <option value="">All Sides</option><option value="long">Long</option><option value="short">Short</option>
  </select>
  <select id="filterFlag" onchange="applyFilters()">
    <option value="">All Flags</option>
    <option value="confirmed">Has Confirmed</option>
    <option value="suggested">Has Suggested</option>
    <option value="accuracy">Has Accuracy</option>
  </select>
  <div style="flex:1"></div>
  <span style="font-size:12px;color:#666" id="countLabel"></span>
</div>

<div class="table-wrap">
<table>
<thead>
<tr>
  <th onclick="sortBy('package_name')">Package</th>
  <th onclick="sortBy('symbol')">Symbol</th>
  <th onclick="sortBy('timeframe')">TF</th>
  <th onclick="sortBy('side')">Side</th>
  <th onclick="sortBy('trade_count')">Trades</th>
  <th>Flags</th>
  <th>Tags</th>
  <th onclick="sortBy('created_at')">Created</th>
  <th>Actions</th>
</tr>
</thead>
<tbody id="tableBody"></tbody>
</table>
<div class="empty" id="emptyMsg" style="display:none">No packages match your filters.</div>
</div>

<div class="footer">Local Chart Overlay — Replay Index</div>

<script>
const ENTRIES = {entries_json_safe};
const SYMBOLS = {symbols_json};
const TIMEFRAMES = {timeframes_json};
let sortKey = 'symbol';
let sortAsc = true;

function init() {{
  // Populate filter dropdowns
  const symSel = document.getElementById('filterSymbol');
  SYMBOLS.forEach(s => {{ const o=document.createElement('option'); o.value=s; o.textContent=s; symSel.appendChild(o); }});
  const tfSel = document.getElementById('filterTf');
  TIMEFRAMES.forEach(t => {{ const o=document.createElement('option'); o.value=t; o.textContent=t; tfSel.appendChild(o); }});
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const sym = document.getElementById('filterSymbol').value;
  const tf = document.getElementById('filterTf').value;
  const side = document.getElementById('filterSide').value;
  const flag = document.getElementById('filterFlag').value;

  let filtered = ENTRIES.filter(e => {{
    if (q && !(e.package_name+' '+e.symbol+' '+e.timeframe+' '+e.side+' '+(e.model||'')+' '+(e.tags||[]).join(' ')+' '+(e.notes||'')).toLowerCase().includes(q)) return false;
    if (sym && e.symbol !== sym) return false;
    if (tf && e.timeframe !== tf) return false;
    if (side && e.side !== side) return false;
    if (flag === 'confirmed' && !e.has_confirmed_schematic) return false;
    if (flag === 'suggested' && !e.has_suggested_schematic) return false;
    if (flag === 'accuracy' && !e.has_accuracy_report) return false;
    return true;
  }});

  filtered.sort((a, b) => {{
    let va = a[sortKey] || '';
    let vb = b[sortKey] || '';
    if (typeof va === 'number' && typeof vb === 'number') return sortAsc ? va - vb : vb - va;
    va = String(va).toLowerCase(); vb = String(vb).toLowerCase();
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});

  renderTable(filtered);
  document.getElementById('countLabel').textContent = filtered.length + ' of ' + ENTRIES.length;
  document.getElementById('emptyMsg').style.display = filtered.length ? 'none' : 'block';
}}

function sortBy(key) {{
  if (sortKey === key) sortAsc = !sortAsc;
  else {{ sortKey = key; sortAsc = true; }}
  applyFilters();
}}

function renderTable(entries) {{
  const tb = document.getElementById('tableBody');
  tb.innerHTML = entries.map(e => {{
    const flags = [];
    if (e.has_confirmed_schematic) flags.push('<span class="pill pill-green">Confirmed</span>');
    if (e.has_suggested_schematic) flags.push('<span class="pill pill-blue">Suggested</span>');
    if (e.has_accuracy_report) flags.push('<span class="pill pill-yellow">Accuracy</span>');
    if (!flags.length) flags.push('<span class="pill pill-gray">None</span>');

    const fileBadges = [];
    if (e.files.has_replay_html) fileBadges.push('<span class="pill pill-green">Replay</span>');
    else fileBadges.push('<span class="pill pill-red">Replay</span>');
    if (e.files.has_overlay_pine) fileBadges.push('<span class="pill pill-green">Pine</span>');
    else fileBadges.push('<span class="pill pill-red">Pine</span>');
    if (e.files.has_open_chart) fileBadges.push('<span class="pill pill-green">Chart</span>');
    if (e.files.has_readme) fileBadges.push('<span class="pill pill-green">Readme</span>');

    const replayClass = e.replay_link ? 'primary' : 'primary disabled';
    const chartClass = e.chart_link ? 'secondary' : 'secondary disabled';
    const replayHref = e.replay_link || '#';
    const chartHref = e.chart_link || '#';

    const created = e.created_at ? e.created_at.replace('T',' ').substring(0,19) : '';
    const ids = e.trade_ids.join(', ');
    const sidePill = e.side === 'long'
      ? '<span class="pill pill-green">LONG</span>'
      : '<span class="pill pill-red">SHORT</span>';

    const tagPills = (e.tags||[]).map(t => '<span class="pill pill-blue">'+esc(t)+'</span>').join('');
    const noteTip = e.notes ? ' title="'+esc(e.notes)+'"' : '';
    const noteIcon = e.notes ? '<span style="cursor:help;color:#ffd54f;margin-left:4px"'+noteTip+'>&#128221;</span>' : '';

    return '<tr>' +
      '<td><strong>'+esc(e.package_name)+'</strong><br><span style="font-size:11px;color:#666">IDs: '+esc(ids)+'</span></td>' +
      '<td>'+esc(e.symbol)+'</td>' +
      '<td>'+esc(e.timeframe)+'</td>' +
      '<td>'+sidePill+'</td>' +
      '<td>'+e.trade_count+'</td>' +
      '<td><div class="badge-row">'+flags.join('')+'</div></td>' +
      '<td><div class="badge-row">'+tagPills+(tagPills?'':' <span style="color:#555">-</span>')+noteIcon+'</div></td>' +
      '<td style="font-size:12px;color:#888">'+esc(created)+'</td>' +
      '<td><a class="link-btn '+replayClass+'" href="'+esc(replayHref)+'">Replay</a>' +
          '<a class="link-btn '+chartClass+'" href="'+esc(chartHref)+'">Pine</a></td>' +
      '</tr>';
  }}).join('');
}}

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

init();
</script>
</body>
</html>"""


def _entry_to_dict(e: ReplayPackageEntry) -> dict:
    """Convert entry to a plain dict for JSON serialization."""
    return {
        "package_name": e.package_name,
        "relative_dir": e.relative_dir,
        "trade_ids": e.trade_ids,
        "trade_count": e.trade_count,
        "symbol": e.symbol,
        "timeframe": e.timeframe,
        "side": e.side,
        "model": e.model,
        "created_at": e.created_at,
        "has_confirmed_schematic": e.has_confirmed_schematic,
        "has_suggested_schematic": e.has_suggested_schematic,
        "has_accuracy_report": e.has_accuracy_report,
        "tags": e.tags,
        "notes": e.notes,
        "files": {
            "has_replay_html": e.files.has_replay_html,
            "has_replay_data": e.files.has_replay_data,
            "has_overlay_pine": e.files.has_overlay_pine,
            "has_open_chart": e.files.has_open_chart,
            "has_readme": e.files.has_readme,
            "has_manifest": e.files.has_manifest,
        },
        "replay_link": e.replay_link,
        "chart_link": e.chart_link,
        "readme_link": e.readme_link,
    }
