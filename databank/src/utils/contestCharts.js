export function drawContestScatter(canvas, data, performanceGflops) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(280, rect.width || 640);
  const height = 340;
  const devicePixelRatio = window.devicePixelRatio || 1;
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const padding = { left: 68, right: 18, top: 20, bottom: 44 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xValues = data.map((row) => Math.max(1, Number.isFinite(row.nnz) ? row.nnz : 1));
  const yValues = data.map(performanceGflops);
  const rawXMinimum = Math.min(...(xValues.length ? xValues : [1]));
  const rawXMaximum = Math.max(...(xValues.length ? xValues : [10]));
  const xLogMinimum = Math.floor(Math.log10(rawXMinimum));
  const xLogMaximum = Math.max(xLogMinimum + 1, Math.ceil(Math.log10(rawXMaximum)));
  const xMinimum = 10 ** xLogMinimum;
  const yMaximum = Math.max(...(yValues.length ? yValues : [1])) * 1.15;
  const xScale = (value) => padding.left
    + (Math.log10(Math.max(xMinimum, value)) - xLogMinimum) / (xLogMaximum - xLogMinimum) * plotWidth;
  const yScale = (value) => padding.top + plotHeight - Math.max(0, value) / yMaximum * plotHeight;

  context.strokeStyle = "#e8edf2";
  context.fillStyle = "#647180";
  context.font = "12px system-ui, sans-serif";
  for (let gridlineIndex = 0; gridlineIndex <= 4; gridlineIndex += 1) {
    const y = padding.top + plotHeight * gridlineIndex / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
    context.fillStyle = "#647180";
    context.textAlign = "right";
    context.fillText(formatGflops(yMaximum * (4 - gridlineIndex) / 4), padding.left - 8, y + 4);
  }

  const exponentSpan = xLogMaximum - xLogMinimum;
  const tickExponents = exponentSpan <= 6
    ? Array.from({ length: exponentSpan + 1 }, (_, tickIndex) => xLogMinimum + tickIndex)
    : Array.from({ length: 5 }, (_, tickIndex) => Math.round(xLogMinimum + exponentSpan * tickIndex / 4));
  for (const exponent of tickExponents) {
    const x = xScale(10 ** exponent);
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, padding.top + plotHeight);
    context.stroke();
    context.textAlign = "center";
    context.fillText(formatNnz(10 ** exponent), x, height - 27);
  }

  context.strokeStyle = "#aeb9c5";
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();
  context.textAlign = "center";
  context.fillText("matrix nnz (log)", padding.left + plotWidth / 2, height - 12);
  context.save();
  context.translate(14, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("GFLOP/s", 0, 0);
  context.restore();

  for (const row of data) {
    context.fillStyle = "rgba(37, 77, 158, 0.48)";
    context.beginPath();
    context.arc(xScale(row.nnz), yScale(performanceGflops(row)), 4, 0, Math.PI * 2);
    context.fill();
  }
  if (!data.length) {
    context.fillStyle = "#647180";
    context.textAlign = "center";
    context.fillText("No published cases", padding.left + plotWidth / 2, padding.top + plotHeight / 2);
  }
}

function formatNnz(value) {
  if (value >= 1e6) return `${(value / 1e6).toPrecision(3)}M`;
  if (value >= 1e3) return `${(value / 1e3).toPrecision(3)}K`;
  return Math.round(value).toString();
}

function formatGflops(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.01) return value.toFixed(3);
  return value.toExponential(1);
}
