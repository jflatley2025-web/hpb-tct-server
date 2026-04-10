"""HTML launcher template — self-contained, no external dependencies."""
from __future__ import annotations

import html


def render_html(
    pine_script: str,
    title: str = "TradingView Overlay Loader",
    symbol: str = "",
    timeframe: str = "",
    trade_count: int = 1,
    generated_at: str = "",
) -> str:
    """Render a self-contained HTML file with embedded Pine Script.

    The HTML provides:
      - "Copy Pine Script" button (clipboard API + textarea fallback)
      - "Open TradingView" link
      - Step-by-step instructions

    Args:
        pine_script: Raw Pine Script content to embed.
        title: Page title.
        symbol: Symbol label for display.
        timeframe: Timeframe label for display.
        trade_count: Number of trades in the script.
        generated_at: Generation timestamp string.

    Returns:
        Complete HTML string.
    """
    # Escape the Pine script for safe embedding inside a JS template literal.
    # We need to escape backticks, backslashes, and ${} to prevent
    # JS template literal injection. Then HTML-escape for the <textarea>.
    pine_js_safe = (
        pine_script
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )
    pine_html_safe = html.escape(pine_script)

    subtitle_parts = []
    if symbol:
        subtitle_parts.append(symbol)
    if timeframe:
        subtitle_parts.append(timeframe)
    subtitle_parts.append(f"{trade_count} trade{'s' if trade_count != 1 else ''}")
    subtitle = " | ".join(subtitle_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 640px;
    width: 100%;
  }}
  h1 {{
    font-size: 24px;
    color: #ffffff;
    margin-bottom: 4px;
  }}
  .subtitle {{
    font-size: 14px;
    color: #8888aa;
    margin-bottom: 32px;
  }}
  .btn-group {{
    display: flex;
    gap: 12px;
    margin-bottom: 32px;
  }}
  .btn {{
    padding: 14px 28px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .btn-primary {{
    background: #2962ff;
    color: #ffffff;
  }}
  .btn-primary:hover {{
    background: #1e4fd6;
    transform: translateY(-1px);
  }}
  .btn-secondary {{
    background: #2a2a4a;
    color: #c0c0e0;
    border: 1px solid #3a3a5a;
  }}
  .btn-secondary:hover {{
    background: #3a3a5a;
    transform: translateY(-1px);
  }}
  .btn-success {{
    background: #26a69a;
    color: #ffffff;
  }}
  .status {{
    height: 24px;
    font-size: 14px;
    color: #26a69a;
    margin-bottom: 32px;
  }}
  .instructions {{
    background: #16162a;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .instructions h2 {{
    font-size: 16px;
    color: #ffffff;
    margin-bottom: 16px;
  }}
  .instructions ol {{
    padding-left: 20px;
  }}
  .instructions li {{
    margin-bottom: 10px;
    line-height: 1.5;
    color: #b0b0cc;
  }}
  .instructions li strong {{
    color: #e0e0ff;
  }}
  .script-preview {{
    background: #0d0d1a;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 16px;
  }}
  .script-preview summary {{
    cursor: pointer;
    font-size: 13px;
    color: #6666aa;
    margin-bottom: 8px;
  }}
  .script-preview textarea {{
    width: 100%;
    height: 200px;
    background: #0d0d1a;
    color: #aaaacc;
    border: none;
    font-family: "Fira Code", "Consolas", monospace;
    font-size: 12px;
    resize: vertical;
    outline: none;
  }}
  .footer {{
    margin-top: 32px;
    font-size: 11px;
    color: #444466;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">{html.escape(subtitle)}</p>

  <div class="btn-group">
    <button class="btn btn-primary" id="copyBtn" onclick="copyScript()">
      Copy Pine Script
    </button>
    <a class="btn btn-secondary" href="https://www.tradingview.com/chart/"
       target="_blank" rel="noopener">
      Open TradingView
    </a>
  </div>

  <div class="status" id="status"></div>

  <div class="instructions">
    <h2>How to use</h2>
    <ol>
      <li>Click <strong>Copy Pine Script</strong> above</li>
      <li>Click <strong>Open TradingView</strong> (or go to your chart)</li>
      <li>Open the <strong>Pine Editor</strong> (bottom panel)</li>
      <li><strong>Paste</strong> the script (Ctrl+V / Cmd+V)</li>
      <li>Click <strong>Add to chart</strong></li>
    </ol>
  </div>

  <div class="script-preview">
    <details>
      <summary>View Pine Script source</summary>
      <textarea id="scriptArea" readonly>{pine_html_safe}</textarea>
    </details>
  </div>

  <p class="footer">Generated {html.escape(generated_at)} | Local Chart Overlay</p>
</div>

<script>
const PINE_SCRIPT = `{pine_js_safe}`;

async function copyScript() {{
  const btn = document.getElementById('copyBtn');
  const status = document.getElementById('status');
  try {{
    await navigator.clipboard.writeText(PINE_SCRIPT);
    btn.textContent = 'Copied!';
    btn.className = 'btn btn-success';
    status.textContent = 'Pine script copied to clipboard';
    setTimeout(() => {{
      btn.textContent = 'Copy Pine Script';
      btn.className = 'btn btn-primary';
      status.textContent = '';
    }}, 3000);
  }} catch (err) {{
    // Fallback: select the textarea content
    const ta = document.getElementById('scriptArea');
    ta.parentElement.open = true;
    ta.select();
    try {{
      document.execCommand('copy');
      status.textContent = 'Copied via fallback method';
    }} catch (e) {{
      status.textContent = 'Copy failed. Please select and copy manually from below.';
    }}
  }}
}}
</script>
</body>
</html>"""
