# Local Chart Overlay

A completely isolated sidecar module that reads existing trade data and generates stable, non-recalculating TradingView chart overlays.

**Zero coupling** to the existing trading system. Treats all external data as read-only.

## Quick Start

```bash
# Install dependencies (isolated from main project)
pip install -r local_chart_overlay/requirements.txt

# Import trades from the live trade log
python -m local_chart_overlay import-json ../schematics_5b_trade_log.json

# List imported trades
python -m local_chart_overlay list-trades

# View a specific trade
python -m local_chart_overlay show-trade 1

# Attach schematic data interactively
python -m local_chart_overlay attach-schematic 1

# Export Pine Script for all trades
python -m local_chart_overlay export-pine --all

# Export specific trades
python -m local_chart_overlay export-pine 1 2 3
```

## Architecture

```
Ingest (JSON/CSV) --> SQLite (single source of truth) --> Pine Script (static render)
```

### Layer 1: Ingestion
Reads trade data from JSON or CSV. No schematic data expected at this stage.

### Layer 2: Frozen Schematic Store (SQLite)
Stores immutable time + price anchors. Once saved, never recalculated. This is the single source of truth.

### Layer 3: Pine Script Renderer
Generates `.pine` files with all data baked into static arrays. Pine does ZERO calculation.

## Pine Script Usage

1. Run `export-pine` to generate `.pine` files in `exports/`
2. Open TradingView Desktop or web
3. Open Pine Editor
4. Paste the contents of the generated `.pine` file
5. Add to chart
6. Use the "Trade #" input to select which trade to display

All drawings use `xloc.bar_time` with Unix milliseconds. Switching timeframes will NOT shift or break the overlay.

## CLI Commands

| Command | Description |
|---------|-------------|
| `import-json <file>` | Import from JSON trade log |
| `import-csv <file>` | Import from CSV (configurable columns) |
| `list-trades` | List all trades with filters |
| `show-trade <id>` | Show full trade + schematic details |
| `attach-schematic <id>` | Interactively add/edit schematic anchors |
| `export-pine <ids...>` | Generate Pine Script overlay |

## Data Storage

SQLite database at `data/overlay.db` with three tables:
- `trades` — normalized trade records
- `schematics` — frozen schematic anchors (1:1 with trades)
- `render_exports` — export audit trail

## Testing

```bash
cd local_chart_overlay
python -m pytest tests/ -v
```
