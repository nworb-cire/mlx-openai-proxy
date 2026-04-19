from __future__ import annotations


def dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MLX Proxy Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #12161d;
      --panel: #1a2230;
      --muted: #8da1b9;
      --text: #edf3fb;
      --accent: #7ad3ff;
      --good: #8ef0b5;
      --bad: #ff8f8f;
      --border: #2d3a4d;
    }
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: radial-gradient(circle at top, #20304a 0%, var(--bg) 42%);
      color: var(--text);
    }
    .wrap {
      max-width: 1450px;
      margin: 0 auto;
      padding: 24px;
    }
    h1, h2 { margin: 0 0 12px; }
    .meta { color: var(--muted); margin-bottom: 24px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }
    .card, .panel {
      background: rgba(26, 34, 48, 0.92);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .graphs {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .value { font-size: 28px; margin-top: 8px; }
    .chart-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .chart-title {
      font-size: 15px;
      font-weight: 700;
    }
    .chart-sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .chart-stat {
      color: var(--accent);
      font-size: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }
    .chart-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .chart-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
      flex: 0 0 auto;
    }
    .chart {
      width: 100%;
      height: 180px;
      display: block;
      background:
        linear-gradient(to bottom, rgba(141, 161, 185, 0.08), rgba(141, 161, 185, 0.02)),
        linear-gradient(to right, rgba(141, 161, 185, 0.06) 1px, transparent 1px);
      background-size: 100% 100%, 14.285% 100%;
      border-radius: 10px;
      overflow: hidden;
    }
    .chart-wrap {
      position: relative;
    }
    .chart-tooltip {
      position: absolute;
      pointer-events: none;
      z-index: 2;
      display: none;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(18, 22, 29, 0.96);
      color: var(--text);
      font-size: 12px;
      white-space: nowrap;
      box-shadow: 0 6px 18px rgba(0,0,0,0.25);
      transform: translate(10px, -10px);
    }
    .chart-hit {
      fill: transparent;
      cursor: default;
    }
    .chart-empty {
      height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
      border: 1px dashed var(--border);
      border-radius: 10px;
    }
    .chart-footer {
      margin-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 10px 8px;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .mono { white-space: nowrap; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .muted { color: var(--muted); }
    @media (max-width: 1100px) {
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .graphs { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MLX Proxy Dashboard</h1>
    <div class="meta">Live queue plus persisted size metadata. Active token counters are approximate during decode. History charts use overall service time and now plot one line per model. No prompt or completion text is stored.</div>
    <div class="cards">
      <div class="card"><div class="label">Active Requests</div><div class="value" id="active-count">0</div></div>
      <div class="card"><div class="label">Completed</div><div class="value" id="completed-count">0</div></div>
      <div class="card"><div class="label">Errors</div><div class="value" id="error-count">0</div></div>
      <div class="card"><div class="label">Avg Service / Wait</div><div class="value" id="avg-duration">0 / 0 ms</div></div>
    </div>
    <div class="graphs">
      <div class="panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">Wait Time</div>
            <div class="chart-sub">10-minute rolling average over the last 12 hours</div>
          </div>
          <div class="chart-stat" id="wait-stat">No data</div>
        </div>
        <div id="wait-chart"></div>
      </div>
      <div class="panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">Input Size</div>
            <div class="chart-sub">10-minute rolling average over the last 12 hours</div>
          </div>
          <div class="chart-stat" id="input-size-stat">No data</div>
        </div>
        <div id="input-size-chart"></div>
      </div>
      <div class="panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">Input TPS</div>
            <div class="chart-sub">10-minute rolling average over the last 12 hours</div>
          </div>
          <div class="chart-stat" id="input-tps-stat">No data</div>
        </div>
        <div id="input-tps-chart"></div>
      </div>
      <div class="panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">Output TPS</div>
            <div class="chart-sub">10-minute rolling average over the last 12 hours</div>
          </div>
          <div class="chart-stat" id="output-tps-stat">No data</div>
        </div>
        <div id="output-tps-chart"></div>
      </div>
    </div>
    <div class="panel" style="margin-bottom: 16px;">
      <h2>Active Requests</h2>
      <table>
        <thead>
          <tr>
            <th>Age</th>
            <th>Model</th>
            <th>Status</th>
            <th>Input</th>
            <th>Images</th>
            <th>Schema</th>
            <th>Reasoning</th>
            <th>Output</th>
            <th>Service</th>
            <th>Wait</th>
          </tr>
        </thead>
        <tbody id="running-body"></tbody>
      </table>
    </div>
    <div class="panel" style="margin-bottom: 16px;">
      <h2>Queued Requests</h2>
      <table>
        <thead>
          <tr>
            <th>Age</th>
            <th>Model</th>
            <th>Status</th>
            <th>Input</th>
            <th>Images</th>
            <th>Schema</th>
            <th>Reasoning</th>
            <th>Output</th>
            <th>Service</th>
            <th>Wait</th>
          </tr>
        </thead>
        <tbody id="queued-body"></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>History</h2>
      <table>
        <thead>
          <tr>
            <th>Started</th>
            <th>Model</th>
            <th>Status</th>
            <th>Input</th>
            <th>Images</th>
            <th>Schema</th>
            <th>Reasoning</th>
            <th>Output</th>
            <th>Service</th>
            <th>Wait</th>
          </tr>
        </thead>
        <tbody id="history-body"></tbody>
      </table>
    </div>
  </div>
  <script>
    const CHART_PALETTES = {
      wait: ['#ffb36b', '#ffe08a', '#ffd1a1', '#ffc16b'],
      inputSize: ['#7ad3ff', '#53b7f8', '#9ee7ff', '#67c8dd'],
      outputTps: ['#8ef0b5', '#55d98d', '#b5f7cd', '#6cc59d'],
      inputTps: ['#ff8f8f', '#ff6b6b', '#ffc0c0', '#ff9f6b'],
    };
    function fmtAge(ms) {
      if (ms == null) return '';
      const sec = Math.floor(ms / 1000);
      return sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m ${sec % 60}s`;
    }
    function fmtTime(ts) {
      if (!ts) return '';
      return new Date(ts * 1000).toLocaleTimeString();
    }
    function row(cells) {
      return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join('')}</tr>`;
    }
    function fmtNumber(value, digits = 1) {
      if (value == null || !Number.isFinite(value)) return '';
      return Number(value).toFixed(digits);
    }
    function modelLabel(model) {
      return model || 'unknown';
    }
    function escapeHtml(text) {
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function buildSeriesByModel(items, pickY) {
      const byModel = new Map();
      items.forEach((item) => {
        const model = modelLabel(item.model);
        const point = { x: item.started_at, y: pickY(item), model };
        if (!point.x || point.y == null || !Number.isFinite(point.y)) return;
        if (!byModel.has(model)) byModel.set(model, []);
        byModel.get(model).push(point);
      });
      return Array.from(byModel.entries()).map(([model, points]) => ({
        model,
        points: points.sort((a, b) => a.x - b.x),
      }));
    }
    const HISTORY_WINDOW_SECONDS = 12 * 60 * 60;
    const ROLLING_WINDOW_SECONDS = 10 * 60;
    const ROLLING_STEP_SECONDS = 10 * 60;
    function buildRollingSeriesByModel(items, pickY, nowSeconds) {
      const rawSeries = buildSeriesByModel(items, pickY);
      const startSeconds = nowSeconds - HISTORY_WINDOW_SECONDS;
      const firstSample = Math.ceil(startSeconds / ROLLING_STEP_SECONDS) * ROLLING_STEP_SECONDS;
      return rawSeries.map((entry) => {
        const points = [];
        let left = 0;
        let right = 0;
        for (let sample = firstSample; sample <= nowSeconds; sample += ROLLING_STEP_SECONDS) {
          while (left < entry.points.length && entry.points[left].x < sample - ROLLING_WINDOW_SECONDS) {
            left += 1;
          }
          while (right < entry.points.length && entry.points[right].x <= sample) {
            right += 1;
          }
          const windowPoints = entry.points.slice(left, right);
          if (!windowPoints.length) continue;
          const avg = windowPoints.reduce((sum, point) => sum + point.y, 0) / windowPoints.length;
          points.push({ x: sample, y: avg, model: entry.model });
        }
        return { model: entry.model, points };
      }).filter((entry) => entry.points.length);
    }
    function renderChart(containerId, statId, series, options) {
      const container = document.getElementById(containerId);
      const stat = document.getElementById(statId);
      if (!series.length) {
        container.innerHTML = '<div class="chart-empty">No data yet</div>';
        stat.textContent = 'No data';
        return;
      }
      const allPoints = series.flatMap((entry) => entry.points);
      const width = 640;
      const height = 180;
      const padL = 14;
      const padR = 10;
      const padT = 12;
      const padB = 24;
      const xMin = options.xMin != null ? options.xMin : Math.min(...allPoints.map((point) => point.x));
      const xMax = options.xMax != null ? options.xMax : Math.max(...allPoints.map((point) => point.x));
      const yMin = 0;
      const yMaxRaw = options.yMax != null
        ? options.yMax
        : Math.max(...allPoints.map((point) => point.y));
      const yMax = yMaxRaw > 0 ? yMaxRaw : 1;
      const xSpan = Math.max(1, xMax - xMin);
      const ySpan = Math.max(1, yMax - yMin);
      const xPos = (x) => padL + ((x - xMin) / xSpan) * (width - padL - padR);
      const yPos = (y) => height - padB - ((y - yMin) / ySpan) * (height - padT - padB);
      const palette = CHART_PALETTES[options.palette] || CHART_PALETTES.wait;
      stat.innerHTML = series.map((entry, index) => {
        const color = palette[index % palette.length];
        const latest = entry.points[entry.points.length - 1].y;
        return `<span class="chart-chip"><span class="chart-dot" style="background:${color}"></span>${escapeHtml(entry.model)} ${escapeHtml(options.format(latest))}</span>`;
      }).join('');
      container.innerHTML = `
        <div class="chart-wrap">
          <svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${options.title}">
            <line x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" stroke="rgba(141,161,185,0.35)" stroke-width="1" />
            <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${height - padB}" stroke="rgba(141,161,185,0.25)" stroke-width="1" />
            ${series.map((entry, index) => {
              const color = palette[index % palette.length];
              const polyline = entry.points.map((point) => `${xPos(point.x)},${yPos(point.y)}`).join(' ');
              const area = `${padL},${height - padB} ${polyline} ${xPos(entry.points[entry.points.length - 1].x)},${height - padB}`;
              return `
                <polygon points="${area}" fill="${color}" opacity="0.08"></polygon>
                <polyline points="${polyline}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>
                ${entry.points.map((point) => `<circle cx="${xPos(point.x)}" cy="${yPos(point.y)}" r="2.5" fill="${color}"></circle>`).join('')}
                ${entry.points.map((point) => `<circle class="chart-hit" data-label="${escapeHtml(entry.model)} ${fmtTime(point.x)}: ${escapeHtml(options.format(point.y))}" cx="${xPos(point.x)}" cy="${yPos(point.y)}" r="12"></circle>`).join('')}
              `;
            }).join('')}
          </svg>
          <div class="chart-tooltip"></div>
          <div class="chart-footer">
            <span>${fmtTime(xMin)}</span>
            <span>${fmtTime(xMax)}</span>
          </div>
        </div>
      `;
      const tooltip = container.querySelector('.chart-tooltip');
      const wrap = container.querySelector('.chart-wrap');
      container.querySelectorAll('.chart-hit').forEach((node) => {
        node.addEventListener('mouseenter', () => {
          tooltip.textContent = node.dataset.label || '';
          tooltip.style.display = 'block';
        });
        node.addEventListener('mousemove', (event) => {
          const rect = wrap.getBoundingClientRect();
          tooltip.style.left = `${event.clientX - rect.left}px`;
          tooltip.style.top = `${event.clientY - rect.top}px`;
        });
        node.addEventListener('mouseleave', () => {
          tooltip.style.display = 'none';
        });
      });
    }
    async function refresh() {
      const [summaryRes, activeRes, historyRes] = await Promise.all([
        fetch('/admin/api/summary'),
        fetch('/admin/api/active'),
        fetch('/admin/api/history?limit=1000'),
      ]);
      const summary = await summaryRes.json();
      const active = await activeRes.json();
      const history = await historyRes.json();
      const nowSeconds = Date.now() / 1000;
      const cutoffSeconds = nowSeconds - HISTORY_WINDOW_SECONDS;
      const running = active.items.filter((item) => item.state === 'running');
      const queued = active.items.filter((item) => item.state === 'queued');
      const recentHistory = history.items
        .filter((item) => (item.started_at || 0) >= cutoffSeconds);
      const completed = recentHistory
        .filter((item) => item.status === 'completed')
        .sort((a, b) => (a.started_at || 0) - (b.started_at || 0));
      const waitSeries = buildRollingSeriesByModel(
        completed,
        (item) => item.queue_duration_ms != null ? item.queue_duration_ms / 1000 : null,
        nowSeconds,
      );
      const inputSizeSeries = buildRollingSeriesByModel(
        completed,
        (item) => item.prompt_tokens ?? null,
        nowSeconds,
      );
      const inputTpsSeries = buildRollingSeriesByModel(
        completed,
        (item) => item.service_duration_ms && item.prompt_tokens != null
          ? (item.prompt_tokens * 1000) / item.service_duration_ms
          : null,
        nowSeconds,
      );
      const outputTpsSeries = buildRollingSeriesByModel(
        completed,
        (item) => item.service_duration_ms && item.completion_tokens != null
          ? (item.completion_tokens * 1000) / item.service_duration_ms
          : null,
        nowSeconds,
      );
      const sharedTpsMax = Math.max(
        1,
        ...outputTpsSeries.flatMap((entry) => entry.points.map((point) => point.y)),
        ...inputTpsSeries.flatMap((entry) => entry.points.map((point) => point.y)),
      );

      document.getElementById('active-count').textContent = summary.active_count;
      document.getElementById('completed-count').textContent = summary.completed_count;
      document.getElementById('error-count').textContent = summary.error_count;
      document.getElementById('avg-duration').textContent = `${summary.avg_service_duration_ms} / ${summary.avg_queue_duration_ms} ms`;

      renderChart(
        'wait-chart',
        'wait-stat',
        waitSeries,
        {
          title: 'Wait time over time',
          palette: 'wait',
          format: (value) => `${fmtNumber(value, 1)}s`,
          xMin: cutoffSeconds,
          xMax: nowSeconds,
        }
      );
      renderChart(
        'input-size-chart',
        'input-size-stat',
        inputSizeSeries,
        {
          title: 'Input size over time',
          palette: 'inputSize',
          format: (value) => `${Math.round(value)} tok`,
          xMin: cutoffSeconds,
          xMax: nowSeconds,
        }
      );
      renderChart(
        'input-tps-chart',
        'input-tps-stat',
        inputTpsSeries,
        {
          title: 'Input TPS over time',
          palette: 'inputTps',
          format: (value) => `${fmtNumber(value, 1)} t/s`,
          yMax: sharedTpsMax,
          xMin: cutoffSeconds,
          xMax: nowSeconds,
        }
      );
      renderChart(
        'output-tps-chart',
        'output-tps-stat',
        outputTpsSeries,
        {
          title: 'Output TPS over time',
          palette: 'outputTps',
          format: (value) => `${fmtNumber(value, 1)} t/s`,
          yMax: sharedTpsMax,
          xMin: cutoffSeconds,
          xMax: nowSeconds,
        }
      );

      document.getElementById('running-body').innerHTML = running.length
        ? running.map((item) => row([
            fmtAge(item.age_ms),
            item.model || '',
            '<span class="good">running</span>',
            `${item.input_chars || 0} chars / ${item.input_messages || 0} msgs`,
            item.input_image_count || 0,
            item.has_schema ? 'yes' : 'no',
            `${item.live_reasoning_chars || 0} chars / ~${item.live_reasoning_tokens_est || 0} tok`,
            `${item.live_output_chars || 0} chars / ~${item.live_output_tokens_est || 0} tok`,
            `<span class="mono">${fmtAge(item.service_ms)}</span>`,
            `<span class="mono">${fmtAge(item.queue_ms)}</span>`,
          ])).join('')
        : '<tr><td colspan="10" class="muted">No active requests</td></tr>';

      document.getElementById('queued-body').innerHTML = queued.length
        ? queued.map((item) => row([
            fmtAge(item.age_ms),
            item.model || '',
            '<span class="muted">queued</span>',
            `${item.input_chars || 0} chars / ${item.input_messages || 0} msgs`,
            item.input_image_count || 0,
            item.has_schema ? 'yes' : 'no',
            item.asks_for_reasoning ? 'yes' : 'no',
            '',
            '',
            `<span class="mono">${fmtAge(item.queue_ms)}</span>`,
          ])).join('')
        : '<tr><td colspan="10" class="muted">No queued requests</td></tr>';

      document.getElementById('history-body').innerHTML = recentHistory.length
        ? recentHistory.map((item) => row([
            fmtTime(item.started_at),
            item.model || '',
            item.status === 'completed' ? `<span class="good">${item.status}</span>` : `<span class="${item.status === 'error' ? 'bad' : 'muted'}">${item.status}</span>`,
            `${item.input_chars || 0} chars / ${item.input_messages || 0} msgs`,
            item.input_image_count || 0,
            item.has_schema ? 'yes' : 'no',
            item.reasoning_tokens != null || item.reasoning_chars != null
              ? `${item.reasoning_chars ?? 0} chars / ${item.reasoning_tokens ?? 0} tok`
              : '',
            item.completion_tokens != null || item.output_chars != null
              ? `${item.output_chars ?? 0} chars / ${item.completion_tokens ?? 0} tok`
              : '',
            `<span class="mono">${fmtAge(item.service_duration_ms)}</span>`,
            `<span class="mono">${fmtAge(item.queue_duration_ms)}</span>`,
          ])).join('')
        : '<tr><td colspan="10" class="muted">No history yet</td></tr>';
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""
