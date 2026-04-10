"""HTML replay template — self-contained trade lifecycle inspector."""
from __future__ import annotations

import html
import json


def render_replay_html(
    payload_json: str,
    title: str = "",
    pine_script: str = "",
) -> str:
    """Render a complete self-contained HTML replay inspector.

    Args:
        payload_json: JSON string of the ReplayPayload.
        title: Page title override.
        pine_script: Raw Pine Script to embed for TradingView launcher.

    Returns:
        Complete HTML string with embedded JS/CSS/SVG chart.
    """
    safe_json = payload_json.replace("</", "<\\/")
    title_escaped = html.escape(title) if title else "Trade Replay Inspector"

    # Escape Pine script for JS template literal
    pine_js_safe = (
        pine_script
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
        .replace("</", "<\\/")
    ) if pine_script else ""
    pine_html_safe = html.escape(pine_script) if pine_script else ""
    has_pine = "true" if pine_script else "false"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_escaped}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#0f0f1a; color:#d0d0e0; }}
.app {{ display:grid; grid-template-columns:1fr 320px; grid-template-rows:auto auto 1fr; height:100vh; }}
.header {{ grid-column:1/-1; background:#16162a; border-bottom:1px solid #2a2a4a; padding:12px 20px; display:flex; align-items:center; justify-content:space-between; }}
.header h1 {{ font-size:18px; color:#fff; }}
.header .meta {{ font-size:13px; color:#8888aa; }}
.controls {{ grid-column:1/-1; background:#12122a; border-bottom:1px solid #2a2a4a; padding:10px 20px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.stage-btn {{ padding:6px 14px; border:1px solid #3a3a5a; border-radius:6px; background:#1a1a3a; color:#a0a0c0; cursor:pointer; font-size:13px; transition:all .12s; }}
.stage-btn:hover {{ background:#2a2a5a; color:#e0e0ff; }}
.stage-btn.active {{ background:#2962ff; border-color:#2962ff; color:#fff; }}
.nav-btn {{ padding:6px 16px; border:1px solid #3a3a5a; border-radius:6px; background:#1a1a3a; color:#c0c0e0; cursor:pointer; font-size:14px; font-weight:600; }}
.nav-btn:hover {{ background:#2a2a5a; }}
.nav-btn:disabled {{ opacity:.3; cursor:default; }}
.spacer {{ flex:1; }}
.toggle {{ font-size:12px; color:#8888aa; cursor:pointer; user-select:none; display:flex; align-items:center; gap:4px; }}
.toggle input {{ accent-color:#2962ff; }}
.chart-area {{ background:#0d0d1a; overflow:hidden; position:relative; }}
.side-panel {{ background:#12122a; border-left:1px solid #2a2a4a; overflow-y:auto; padding:16px; font-size:13px; }}
.section {{ margin-bottom:20px; }}
.section h3 {{ font-size:14px; color:#fff; margin-bottom:8px; padding-bottom:4px; border-bottom:1px solid #2a2a4a; }}
.kv {{ display:flex; justify-content:space-between; margin-bottom:3px; }}
.kv .k {{ color:#7777aa; }}
.kv .v {{ color:#c0c0e0; text-align:right; }}
.kv .v.win {{ color:#26a69a; }}
.kv .v.loss {{ color:#ef5350; }}
.comp-row {{ background:#16162a; border-radius:4px; padding:6px 8px; margin-bottom:4px; }}
.comp-row .name {{ color:#9999bb; font-size:11px; text-transform:uppercase; margin-bottom:2px; }}
.comp-row .vals {{ display:flex; gap:12px; font-size:12px; }}
.comp-row .hit {{ color:#26a69a; }}
.comp-row .miss {{ color:#ef5350; }}
.breakdown {{ font-size:11px; color:#7777aa; margin-top:2px; }}
.acc-bar {{ height:6px; border-radius:3px; background:#2a2a4a; margin:6px 0; }}
.acc-fill {{ height:100%; border-radius:3px; background:#26a69a; transition:width .3s; }}
.stage-label {{ position:absolute; top:8px; left:8px; font-size:14px; font-weight:600; color:#ffffff; background:rgba(41,98,255,.7); padding:4px 12px; border-radius:4px; z-index:10; }}
svg text {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.tv-btn {{ padding:6px 14px; border:1px solid #3a3a5a; border-radius:6px; background:#1a1a3a; color:#a0a0c0; cursor:pointer; font-size:12px; font-weight:600; transition:all .12s; display:inline-flex; align-items:center; gap:5px; }}
.tv-btn:hover {{ background:#2962ff; border-color:#2962ff; color:#fff; }}
.tv-btn.copied {{ background:#26a69a; border-color:#26a69a; color:#fff; }}
.tv-btn-group {{ display:flex; gap:6px; }}
.overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.7); z-index:100; justify-content:center; align-items:center; }}
.overlay.active {{ display:flex; }}
.overlay-box {{ background:#16162a; border:1px solid #2a2a4a; border-radius:12px; padding:32px; max-width:440px; width:90%; text-align:center; }}
.overlay-box h2 {{ font-size:18px; color:#fff; margin-bottom:12px; }}
.overlay-box ol {{ text-align:left; padding-left:20px; margin:16px 0; }}
.overlay-box li {{ margin-bottom:8px; color:#b0b0cc; line-height:1.5; }}
.overlay-box li strong {{ color:#e0e0ff; }}
.overlay-btn {{ padding:10px 24px; border:none; border-radius:8px; background:#2962ff; color:#fff; font-size:14px; font-weight:600; cursor:pointer; margin-top:8px; }}
.overlay-btn:hover {{ background:#1e4fd6; }}
.fallback-ta {{ width:100%; height:160px; background:#0d0d1a; color:#aaaacc; border:1px solid #3a3a5a; border-radius:6px; font-family:"Fira Code",Consolas,monospace; font-size:11px; padding:10px; margin:12px 0; resize:vertical; }}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1 id="title">{title_escaped}</h1>
    <span class="meta" id="headerMeta"></span>
    <div class="tv-btn-group" id="tvBtns" style="display:none">
      <button class="tv-btn" id="tvLaunchBtn" onclick="launchTradingView()">View on TradingView</button>
      <button class="tv-btn" id="tvCopyBtn" onclick="copyPineOnly()">Copy Pine</button>
    </div>
  </div>
  <div class="overlay" id="tvOverlay">
    <div class="overlay-box">
      <h2 id="overlayTitle">TradingView Ready</h2>
      <div id="overlayContent">
        <ol>
          <li>Pine script <strong>copied to clipboard</strong></li>
          <li>Open the <strong>Pine Editor</strong> in TradingView</li>
          <li><strong>Paste</strong> (Ctrl+V / Cmd+V)</li>
          <li>Click <strong>Add to chart</strong></li>
        </ol>
      </div>
      <div id="overlayFallback" style="display:none">
        <p style="color:#b0b0cc;margin-bottom:8px">Clipboard unavailable. Copy manually:</p>
        <textarea class="fallback-ta" id="fallbackTa" readonly>{pine_html_safe}</textarea>
      </div>
      <button class="overlay-btn" onclick="closeOverlay()">Got it</button>
    </div>
  </div>
  <div class="controls">
    <button class="nav-btn" id="prevBtn" onclick="prevStage()">Prev</button>
    <div id="stageBtns"></div>
    <button class="nav-btn" id="nextBtn" onclick="nextStage()">Next</button>
    <div class="spacer"></div>
    <label class="toggle"><input type="checkbox" id="togConfirmed" checked onchange="render()"> Confirmed</label>
    <label class="toggle"><input type="checkbox" id="togSuggested" onchange="render()"> Suggested</label>
    <label class="toggle"><input type="checkbox" id="togLabels" checked onchange="render()"> Labels</label>
  </div>
  <div class="chart-area" id="chartArea">
    <div class="stage-label" id="stageLabel"></div>
    <svg id="chart" width="100%" height="100%"></svg>
  </div>
  <div class="side-panel" id="sidePanel"></div>
</div>

<script>
const DATA = {safe_json};
const HAS_PINE = {has_pine};
const PINE_SCRIPT = HAS_PINE ? `{pine_js_safe}` : "";
let currentStage = 0;
const PAD = {{ top:30, right:20, bottom:30, left:70 }};

// ── Init ──────────────────────────────────────────────────────────
function init() {{
  const s = DATA.summary;
  document.getElementById('headerMeta').textContent =
    s.symbol+' | '+s.timeframe+' | '+s.side+' | '+s.model;
  document.getElementById('title').textContent =
    'Trade #'+DATA.trade_id+' Replay';

  // Stage buttons
  const cont = document.getElementById('stageBtns');
  DATA.stages.forEach(st => {{
    const b = document.createElement('button');
    b.className = 'stage-btn';
    b.textContent = st.label;
    b.onclick = () => setStage(st.id);
    b.dataset.stage = st.id;
    cont.appendChild(b);
  }});

  if (!DATA.has_suggestion) {{
    document.getElementById('togSuggested').parentElement.style.display='none';
  }}
  if (HAS_PINE) {{
    document.getElementById('tvBtns').style.display='flex';
  }}
  render();
}}

// ── TradingView Launcher ──────────────────────────────────────────
async function launchTradingView() {{
  const btn = document.getElementById('tvLaunchBtn');
  try {{
    await navigator.clipboard.writeText(PINE_SCRIPT);
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    showOverlay(true);
    window.open('https://www.tradingview.com/chart/', '_blank');
    setTimeout(() => {{ btn.textContent = 'View on TradingView'; btn.classList.remove('copied'); }}, 4000);
  }} catch (err) {{
    showOverlay(false);
  }}
}}

async function copyPineOnly() {{
  const btn = document.getElementById('tvCopyBtn');
  try {{
    await navigator.clipboard.writeText(PINE_SCRIPT);
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {{ btn.textContent = 'Copy Pine'; btn.classList.remove('copied'); }}, 3000);
  }} catch (err) {{
    showOverlay(false);
  }}
}}

function showOverlay(clipboardOk) {{
  const overlay = document.getElementById('tvOverlay');
  const content = document.getElementById('overlayContent');
  const fallback = document.getElementById('overlayFallback');
  const title = document.getElementById('overlayTitle');
  if (clipboardOk) {{
    title.textContent = 'TradingView Ready';
    content.style.display = 'block';
    fallback.style.display = 'none';
  }} else {{
    title.textContent = 'Copy Pine Script';
    content.style.display = 'none';
    fallback.style.display = 'block';
    const ta = document.getElementById('fallbackTa');
    ta.select();
  }}
  overlay.classList.add('active');
}}

function closeOverlay() {{
  document.getElementById('tvOverlay').classList.remove('active');
}}

function setStage(n) {{ currentStage=n; render(); }}
function prevStage() {{ if(currentStage>0) {{ currentStage--; render(); }} }}
function nextStage() {{ if(currentStage<DATA.max_stage) {{ currentStage++; render(); }} }}

// ── Render ─────────────────────────────────────────────────────────
function render() {{
  renderChart();
  renderSidePanel();
  // Update stage buttons
  document.querySelectorAll('.stage-btn').forEach(b => {{
    b.classList.toggle('active', parseInt(b.dataset.stage)===currentStage);
  }});
  document.getElementById('stageLabel').textContent =
    DATA.stages[currentStage]?.label || '';
  document.getElementById('prevBtn').disabled = currentStage===0;
  document.getElementById('nextBtn').disabled = currentStage>=DATA.max_stage;
}}

// ── Chart ──────────────────────────────────────────────────────────
function renderChart() {{
  const svg = document.getElementById('chart');
  const area = document.getElementById('chartArea');
  const W = area.clientWidth;
  const H = area.clientHeight;
  svg.setAttribute('viewBox', '0 0 '+W+' '+H);

  const candles = DATA.candles;
  if (!candles.length) {{ svg.innerHTML='<text x="50%" y="50%" text-anchor="middle" fill="#555">No candle data</text>'; return; }}

  const showConf = document.getElementById('togConfirmed').checked;
  const showSugg = document.getElementById('togSuggested').checked;
  const showLabels = document.getElementById('togLabels').checked;

  // Visible candles: progressive reveal based on stage
  let visibleEnd = candles.length;
  if (currentStage < DATA.max_stage) {{
    // Find the latest anchor timestamp at current stage
    const visAnchors = DATA.confirmed_anchors.filter(a => a.visible_from_stage <= currentStage && a.time_ms);
    if (visAnchors.length) {{
      const maxT = Math.max(...visAnchors.map(a => a.time_ms));
      // Show a few bars past the latest anchor
      const idx = candles.findIndex(c => c.time_ms > maxT);
      if (idx > 0) visibleEnd = Math.min(idx + 5, candles.length);
    }}
  }}
  const vis = candles.slice(0, visibleEnd);

  // Price bounds
  let prices = vis.flatMap(c => [c.high, c.low]);
  const visConfAnchors = showConf ? DATA.confirmed_anchors.filter(a => a.visible_from_stage<=currentStage && a.price) : [];
  const visSuggAnchors = showSugg ? DATA.suggested_anchors.filter(a => a.visible_from_stage<=currentStage && a.price) : [];
  prices = prices.concat(visConfAnchors.map(a=>a.price), visSuggAnchors.map(a=>a.price));
  const pMin = Math.min(...prices);
  const pMax = Math.max(...prices);
  const pPad = (pMax-pMin)*0.08 || 1;

  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;
  const barW = Math.max(2, Math.min(12, cW / vis.length * 0.7));
  const gap = cW / vis.length;

  const xOf = i => PAD.left + i * gap + gap/2;
  const yOf = p => PAD.top + cH - ((p - pMin + pPad) / (pMax - pMin + 2*pPad)) * cH;

  let html = '';

  // Grid lines
  const nGrid = 5;
  for (let i=0;i<=nGrid;i++) {{
    const p = pMin - pPad + (pMax - pMin + 2*pPad) * i / nGrid;
    const y = yOf(p);
    html += '<line x1="'+PAD.left+'" y1="'+y+'" x2="'+(W-PAD.right)+'" y2="'+y+'" stroke="#1a1a3a" stroke-width="1"/>';
    html += '<text x="'+(PAD.left-6)+'" y="'+(y+4)+'" text-anchor="end" fill="#555" font-size="10">'+p.toFixed(2)+'</text>';
  }}

  // Candles
  vis.forEach((c, i) => {{
    const x = xOf(i);
    const bull = c.close >= c.open;
    const col = bull ? '#26a69a' : '#ef5350';
    const yH = yOf(c.high), yL = yOf(c.low);
    const yO = yOf(bull ? c.close : c.open);
    const yC = yOf(bull ? c.open : c.close);
    const bodyH = Math.max(1, yC - yO);
    html += '<line x1="'+x+'" y1="'+yH+'" x2="'+x+'" y2="'+yL+'" stroke="'+col+'" stroke-width="1"/>';
    html += '<rect x="'+(x-barW/2)+'" y="'+yO+'" width="'+barW+'" height="'+bodyH+'" fill="'+col+'" rx="1"/>';
  }});

  // Helper: find candle x by timestamp
  const xByTime = (ms) => {{
    if (!ms) return null;
    const idx = vis.findIndex(c => c.time_ms >= ms);
    if (idx >= 0) return xOf(idx);
    if (ms > vis[vis.length-1].time_ms) return xOf(vis.length-1);
    return xOf(0);
  }};

  // Horizontal level lines
  const drawLevel = (anchor, color, dash, labelSide) => {{
    if (!anchor.price) return;
    const y = yOf(anchor.price);
    const x1 = anchor.time_ms ? xByTime(anchor.time_ms) : PAD.left;
    const x2 = W - PAD.right;
    const dashAttr = dash ? ' stroke-dasharray="6,4"' : '';
    html += '<line x1="'+(x1||PAD.left)+'" y1="'+y+'" x2="'+x2+'" y2="'+y+'" stroke="'+color+'" stroke-width="1.5"'+dashAttr+' opacity="0.8"/>';
    if (showLabels) {{
      const lx = labelSide==='right' ? x2+4 : (x1||PAD.left)-4;
      const ta = labelSide==='right' ? 'start' : 'end';
      html += '<text x="'+lx+'" y="'+(y-4)+'" fill="'+color+'" font-size="10" text-anchor="'+ta+'">'+anchor.label+' '+anchor.price.toFixed(2)+'</text>';
    }}
  }};

  // Tap/BOS markers
  const drawMarker = (anchor, color, shape) => {{
    if (!anchor.price || !anchor.time_ms) return;
    const x = xByTime(anchor.time_ms);
    if (!x) return;
    const y = yOf(anchor.price);
    if (shape==='diamond') {{
      html += '<polygon points="'+x+','+(y-8)+' '+(x+6)+','+y+' '+x+','+(y+8)+' '+(x-6)+','+y+'" fill="'+color+'" opacity="0.9"/>';
    }} else {{
      html += '<circle cx="'+x+'" cy="'+y+'" r="5" fill="'+color+'" opacity="0.9"/>';
    }}
    if (showLabels) {{
      html += '<text x="'+x+'" y="'+(y-12)+'" fill="'+color+'" font-size="11" text-anchor="middle" font-weight="600">'+anchor.label.toUpperCase()+'</text>';
    }}
  }};

  // Confirmed anchors
  if (showConf) {{
    const COLORS = {{
      range_high:'#42a5f5', range_eq:'#ffa726', range_low:'#42a5f5',
      tap1:'#ffee58', tap2:'#ffee58', tap3:'#ffee58',
      bos:'#ab47bc',
      entry:'#26a69a', stop_loss:'#ef5350', target:'#26a69a', tp1:'#66bb6a',
      exit:'#ffffff',
    }};
    visConfAnchors.forEach(a => {{
      const col = COLORS[a.label] || '#888';
      if (['range_high','range_eq','range_low','entry','stop_loss','target','tp1'].includes(a.label)) {{
        const dash = ['range_eq','tp1','stop_loss','target'].includes(a.label);
        drawLevel(a, col, dash, 'left');
      }}
      if (['tap1','tap2','tap3'].includes(a.label)) {{
        drawMarker(a, col, 'diamond');
        drawLevel(a, col, true, 'left');
      }}
      if (a.label==='bos') drawMarker(a, col, 'circle');
      if (a.label==='exit') drawMarker(a, '#fff', 'circle');
    }});
  }}

  // Suggested anchors (dashed, alternate colors)
  if (showSugg) {{
    const SCOL = {{
      range_high:'#64b5f6', range_low:'#64b5f6',
      tap1:'#fff176', tap2:'#fff176', tap3:'#fff176',
      bos:'#ce93d8',
    }};
    visSuggAnchors.forEach(a => {{
      const col = SCOL[a.label] || '#999';
      if (['range_high','range_low'].includes(a.label)) {{
        drawLevel(a, col, true, 'right');
      }}
      if (['tap1','tap2','tap3'].includes(a.label)) {{
        if (a.time_ms) {{
          const x = xByTime(a.time_ms);
          const y = yOf(a.price);
          if(x) html += '<rect x="'+(x-5)+'" y="'+(y-5)+'" width="10" height="10" fill="none" stroke="'+col+'" stroke-width="2" stroke-dasharray="3,2" transform="rotate(45,'+x+','+y+')"/>';
          if(showLabels) html += '<text x="'+x+'" y="'+(y+18)+'" fill="'+col+'" font-size="10" text-anchor="middle" font-style="italic">s:'+a.label+'</text>';
        }}
      }}
      if (a.label==='bos' && a.time_ms) {{
        const x = xByTime(a.time_ms);
        const y = yOf(a.price);
        if(x) html += '<circle cx="'+x+'" cy="'+y+'" r="6" fill="none" stroke="'+col+'" stroke-width="2" stroke-dasharray="3,2"/>';
      }}
    }});
  }}

  svg.innerHTML = html;
}}

// ── Side Panel ─────────────────────────────────────────────────────
function renderSidePanel() {{
  const s = DATA.summary;
  let h = '';

  // Trade summary
  h += '<div class="section"><h3>Trade Summary</h3>';
  h += kv('Symbol', s.symbol);
  h += kv('Direction', s.side+' ('+s.direction+')');
  h += kv('Timeframe', s.timeframe);
  h += kv('Model', s.model);
  h += kv('Entry', s.entry_price);
  h += kv('Stop', s.stop_price);
  h += kv('Target', s.target_price);
  if(s.tp1_price) h += kv('TP1', s.tp1_price);
  h += kv('R:R', s.rr || 'n/a');
  h += kv('Score', s.entry_score || 'n/a');
  if(s.htf_bias) h += kv('HTF Bias', s.htf_bias);
  h += kv('Opened', fmtTime(s.opened_at));
  if(s.closed_at) h += kv('Closed', fmtTime(s.closed_at));
  if(s.exit_reason) h += kv('Exit', s.exit_reason);
  const pnlCls = s.is_win ? 'win' : (s.is_win===false ? 'loss' : '');
  if(s.pnl_pct!=null) h += kv('PnL', (s.pnl_pct>0?'+':'')+s.pnl_pct.toFixed(2)+'%', pnlCls);
  if(s.pnl_dollars!=null) h += kv('PnL $', (s.pnl_dollars>0?'+':'')+s.pnl_dollars.toFixed(2), pnlCls);
  h += '</div>';

  // Schematic info
  h += '<div class="section"><h3>Schematic</h3>';
  if(s.schematic_source) {{
    h += kv('Source', s.schematic_source);
    h += kv('Version', s.schematic_version);
    h += kv('Completeness', s.schematic_completeness!=null ? (s.schematic_completeness*100).toFixed(0)+'%' : 'n/a');
  }} else {{
    h += '<div style="color:#666">No schematic attached</div>';
  }}
  h += '</div>';

  // Tags + Notes
  if ((DATA.tags && DATA.tags.length) || DATA.notes) {{
    h += '<div class="section"><h3>Annotations</h3>';
    if (DATA.tags && DATA.tags.length) {{
      h += '<div style="margin-bottom:6px">';
      DATA.tags.forEach(t => {{
        h += '<span class="pill pill-blue" style="margin:2px 3px 2px 0;display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#1565c040;color:#64b5f6;border:1px solid #1976d260">'+esc(t)+'</span>';
      }});
      h += '</div>';
    }}
    if (DATA.notes) {{
      h += '<div style="color:#b0b0cc;font-size:12px;line-height:1.5;padding:6px 0;border-top:1px solid #2a2a4a;margin-top:4px">'+esc(DATA.notes)+'</div>';
    }}
    h += '</div>';
  }}

  // Confirmed anchors visible at current stage
  const visAnchors = DATA.confirmed_anchors.filter(a => a.visible_from_stage <= currentStage);
  if(visAnchors.length) {{
    h += '<div class="section"><h3>Anchors (Stage '+currentStage+')</h3>';
    visAnchors.forEach(a => {{
      h += kv(a.label, a.price ? a.price.toFixed(2) : 'n/a');
    }});
    h += '</div>';
  }}

  // Comparisons
  if (DATA.comparisons.length && document.getElementById('togSuggested').checked) {{
    h += '<div class="section"><h3>Confirmed vs Suggested</h3>';
    DATA.comparisons.forEach(c => {{
      h += '<div class="comp-row">';
      h += '<div class="name">'+c.anchor_name+'</div>';
      h += '<div class="vals">';
      h += '<span>C: '+(c.confirmed_price!=null ? c.confirmed_price.toFixed(2) : 'n/a')+'</span>';
      h += '<span>S: '+(c.suggested_price!=null ? c.suggested_price.toFixed(2) : 'n/a')+'</span>';
      if(c.price_delta_pct!=null) {{
        const cls = c.hit ? 'hit' : 'miss';
        h += '<span class="'+cls+'">'+(c.hit?'HIT':'MISS')+' '+c.price_delta_pct.toFixed(2)+'%</span>';
      }}
      h += '</div>';
      if(c.score_breakdown) {{
        h += '<div class="breakdown">';
        Object.entries(c.score_breakdown).forEach(([k,v]) => {{
          h += k+'='+(v*100).toFixed(0)+'%  ';
        }});
        h += '</div>';
      }}
      h += '</div>';
    }});
    h += '</div>';
  }}

  // Accuracy
  if (DATA.has_accuracy && DATA.accuracy) {{
    const a = DATA.accuracy;
    h += '<div class="section"><h3>Accuracy</h3>';
    h += kv('Hit Rate', (a.hit_rate*100).toFixed(0)+'%');
    h += '<div class="acc-bar"><div class="acc-fill" style="width:'+(a.hit_rate*100)+'%"></div></div>';
    if(a.avg_price_error_pct!=null) h += kv('Avg Price Err', a.avg_price_error_pct.toFixed(3)+'%');
    if(a.avg_time_error_seconds!=null) h += kv('Avg Time Err', (a.avg_time_error_seconds/3600).toFixed(1)+'h');
    if(a.anchor_results) {{
      a.anchor_results.forEach(r => {{
        const cls = r.hit ? 'hit' : 'miss';
        h += '<div class="comp-row"><div class="name">'+r.name+'</div>';
        h += '<div class="vals"><span class="'+cls+'">'+(r.hit?'HIT':'MISS')+'</span>';
        if(r.price_err_pct!=null) h += '<span>'+r.price_err_pct.toFixed(3)+'%</span>';
        h += '</div></div>';
      }});
    }}
    h += '</div>';
  }}

  document.getElementById('sidePanel').innerHTML = h;
}}

function kv(k, v, cls) {{
  return '<div class="kv"><span class="k">'+k+'</span><span class="v'+(cls?' '+cls:'')+'">'+v+'</span></div>';
}}
function fmtTime(iso) {{
  if(!iso) return 'n/a';
  return iso.replace('T',' ').substring(0,19);
}}
function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

window.addEventListener('resize', render);
window.addEventListener('keydown', e => {{
  if(e.key==='ArrowRight'||e.key===' ') {{ e.preventDefault(); nextStage(); }}
  if(e.key==='ArrowLeft') {{ e.preventDefault(); prevStage(); }}
}});
init();
</script>
</body>
</html>"""
