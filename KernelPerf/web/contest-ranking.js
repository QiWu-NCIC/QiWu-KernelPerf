const contestId = decodeURIComponent(window.location.pathname.split("/").filter(Boolean).at(-1));
const title = document.getElementById("contestTitle");
const environment = document.getElementById("contestEnvironment");
const rows = document.getElementById("leaderboardRows");
const empty = document.getElementById("leaderboardEmpty");
const error = document.getElementById("leaderboardError");

function drawContestPlot(canvas, results) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.scale(dpr, dpr);
  const width = rect.width;
  const height = rect.height;
  const pad = { left: 62, right: 14, top: 28, bottom: 36 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const counts = { pass: 0, error: 0, fail: 0 };
  results.forEach((result) => { counts[result.status] = (counts[result.status] || 0) + 1; });
  const colors = { pass: "#2e8b57", error: "#d58a16", fail: "#c94b4b" };
  let barX = pad.left;
  ["pass", "error", "fail"].forEach((status) => {
    const segmentWidth = results.length ? plotWidth * counts[status] / results.length : 0;
    ctx.fillStyle = colors[status];
    ctx.fillRect(barX, 8, segmentWidth, 8);
    barX += segmentWidth;
  });

  const passed = results.filter((result) => result.status === "pass" && result.gflops > 0);
  const xs = results.length ? results.map((result) => Math.max(1, result.nnz)) : [1];
  const ys = passed.length ? passed.map((result) => result.gflops) : [1e-12];
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.max(1e-12, Math.min(...ys) * 0.8);
  const yMax = Math.max(yMin * 10, Math.max(...ys) * 1.25);
  const xScale = (value) => {
    const low = Math.log10(Math.max(1, xMin));
    const high = Math.log10(Math.max(10, xMax));
    return pad.left + (Math.log10(Math.max(1, value)) - low) / Math.max(0.001, high - low) * plotWidth;
  };
  const yScale = (value) => {
    const low = Math.log10(yMin);
    const high = Math.log10(yMax);
    return pad.top + (high - Math.log10(Math.max(yMin, value))) / Math.max(0.001, high - low) * plotHeight;
  };

  ctx.strokeStyle = "#edf1f5";
  for (let index = 0; index <= 3; index += 1) {
    const y = pad.top + index / 3 * plotHeight;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#c8d0da";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  results.forEach((result) => {
    const performance = result.status === "pass" && result.gflops > 0 ? result.gflops : yMin;
    ctx.fillStyle = result.status === "pass" ? "#2563a6" : colors[result.status];
    ctx.beginPath();
    ctx.arc(xScale(result.nnz), yScale(performance), 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#526070";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.fillText(yMax.toPrecision(3), pad.left - 8, pad.top + 4);
  ctx.fillText(yMin.toPrecision(2), pad.left - 8, height - pad.bottom);
  ctx.textAlign = "center";
  ctx.fillText("matrix nnz (log)", pad.left + plotWidth / 2, height - 8);
}

function participantRow(participant) {
  const row = document.createElement("tr");
  const rank = document.createElement("td");
  rank.className = "rank-cell";
  rank.textContent = participant.rank ?? "—";
  const nickname = document.createElement("td");
  nickname.className = "nickname-cell";
  nickname.textContent = participant.nickname;
  const plotCell = document.createElement("td");
  const canvas = document.createElement("canvas");
  canvas.className = "ranking-plot";
  canvas.title = participant.results.map((result) => {
    const performance = result.status === "pass" ? ` · ${result.gflops.toPrecision(6)} GFLOPS` : "";
    return `${result.matrix_name}: ${result.status}${performance} · preprocess ${result.preprocess_ms.toPrecision(6)} ms · solve ${result.runtime_ms.toPrecision(6)} ms`;
  }).join("\n");
  plotCell.appendChild(canvas);
  const score = document.createElement("td");
  score.className = "score-cell";
  score.textContent = participant.score_gflops === null
    ? participant.status
    : participant.score_gflops.toPrecision(6);
  row.append(rank, nickname, plotCell, score);
  requestAnimationFrame(() => drawContestPlot(canvas, participant.results));
  return row;
}

async function refreshLeaderboard() {
  try {
    const response = await fetch(`/api/v1/contests/${encodeURIComponent(contestId)}/leaderboard`);
    if (!response.ok) throw new Error(`Failed to load leaderboard: ${response.status}`);
    const payload = await response.json();
    title.textContent = payload.contest.name;
    document.title = `${payload.contest.name} · KernelPerf`;
    environment.textContent = [
      payload.contest.contest_id,
      payload.contest.backend_id,
      payload.contest.dataset_id,
      payload.contest.suite,
      payload.contest.operator_ids.join(", "),
      new Date(payload.contest.start_time).toLocaleString(),
    ].join(" · ");
    rows.replaceChildren(...payload.participants.map(participantRow));
    empty.hidden = payload.participants.length !== 0;
    error.hidden = true;
  } catch (cause) {
    error.textContent = String(cause);
    error.hidden = false;
  }
}

refreshLeaderboard();
setInterval(refreshLeaderboard, 5000);
