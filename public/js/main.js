const bootstrapElement = document.getElementById("bootstrap-data");
const bootstrapData = JSON.parse(bootstrapElement?.textContent || "{}");

const form = document.getElementById("predict-form");
const teamASelect = document.getElementById("team-a");
const teamBSelect = document.getElementById("team-b");
const calculateButton = document.getElementById("calculate-btn");
const probabilityBars = document.getElementById("probability-bars");
const comparisonBody = document.getElementById("comparison-body");
const toastShell = document.getElementById("toast-shell");
const pcaDetail = document.getElementById("pca-detail");
const clusterList = document.getElementById("cluster-list");

const colors = {
  home: "#35d486",
  draw: "#f0c75e",
  away: "#58a6ff",
};

function formatPercent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setLoading(isLoading) {
  calculateButton?.classList.toggle("is-loading", isLoading);
  if (calculateButton) {
    calculateButton.disabled = isLoading;
  }
}

function showToast(message) {
  if (!toastShell) return;

  const toast = document.createElement("div");
  toast.className = "toast-message";
  toast.textContent = message;
  toastShell.appendChild(toast);

  window.setTimeout(() => {
    toast.remove();
  }, 4200);
}

function renderProbabilityBars(probabilities) {
  probabilityBars.innerHTML = probabilities
    .map(
      (item) => `
        <div class="prob-row">
          <div class="prob-name">${item.flag} ${item.label}</div>
          <div class="prob-track" aria-hidden="true">
            <span class="prob-fill ${item.key}" style="width: ${item.percent}%"></span>
          </div>
          <div class="prob-value">${formatPercent(item.percent)}</div>
        </div>
      `,
    )
    .join("");
}

function renderChart(probabilities) {
  if (!window.Plotly) return;

  const labels = probabilities.map((item) => `${item.flag} ${item.short_label}`);
  const values = probabilities.map((item) => item.percent);
  const barColors = probabilities.map((item) => colors[item.key]);

  const data = [
    {
      type: "bar",
      x: labels,
      y: values,
      text: values.map(formatPercent),
      textposition: "outside",
      cliponaxis: false,
      marker: {
        color: barColors,
        line: {
          color: "rgba(255,255,255,0.18)",
          width: 1,
        },
      },
      hovertemplate: "%{x}<br>%{y:.1f}%<extra></extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 22, r: 12, b: 52, l: 42 },
    yaxis: {
      range: [0, 100],
      gridcolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "#9cb4c9" },
      title: { text: "%", font: { color: "#9cb4c9" } },
    },
    xaxis: {
      tickfont: { color: "#dcecff", size: 12 },
    },
    font: {
      family: "Inter, Segoe UI, sans-serif",
      color: "#eef6ff",
    },
    showlegend: false,
  };

  Plotly.react("probability-chart", data, layout, {
    displayModeBar: false,
    responsive: true,
  });
}

function renderClusterList(pcaMap) {
  if (!clusterList || !pcaMap?.clusters?.length) return;

  clusterList.innerHTML = pcaMap.clusters
    .map(
      (cluster) => `
        <div class="cluster-item">
          <span class="cluster-dot" style="background:${cluster.color}"></span>
          <div>
            <strong>${escapeHtml(cluster.label)}</strong>
            <span>${cluster.count} selecciones</span>
          </div>
          <span class="cluster-score">${Number(cluster.avg_score).toFixed(2)}</span>
        </div>
      `,
    )
    .join("");
}

function renderPcaDetail(point) {
  if (!pcaDetail || !point) return;

  pcaDetail.innerHTML = `
    <div class="pca-detail-head">
      <span class="pca-detail-flag">${escapeHtml(point.flag)}</span>
      <div>
        <span class="eyebrow">${escapeHtml(point.cluster_label)}</span>
        <h4>${escapeHtml(point.display_name)}</h4>
        <p>${escapeHtml(point.profile)} · ${escapeHtml(point.confederation)}</p>
      </div>
    </div>
    <p>Fortalezas principales: ${escapeHtml(point.strengths)}.</p>
    <div class="pca-detail-grid">
      <div><span>Score Poder</span><strong>${Number(point.score).toFixed(3)}</strong></div>
      <div><span>Ranking modelo</span><strong>#${point.model_rank}</strong></div>
      <div><span>Ranking FIFA</span><strong>${escapeHtml(point.fifa_rank)}</strong></div>
      <div><span>Rating Elo</span><strong>${escapeHtml(point.elo)}</strong></div>
      <div><span>Valor mercado</span><strong>${escapeHtml(point.market)}</strong></div>
      <div><span>Forma</span><strong>${escapeHtml(point.form)}</strong></div>
      <div><span>xG / xGA</span><strong>${escapeHtml(point.xg)} / ${escapeHtml(point.xga)}</strong></div>
      <div><span>Lesiones</span><strong>${escapeHtml(point.injuries)}</strong></div>
    </div>
  `;
}

function renderPcaMap(pcaMap) {
  if (!window.Plotly || !pcaMap?.points?.length) return;

  renderClusterList(pcaMap);

  const pointByTeam = new Map(pcaMap.points.map((point) => [point.team, point]));
  const clusterOrder = pcaMap.clusters.map((cluster) => cluster.label);
  const traces = clusterOrder.map((label) => {
    const points = pcaMap.points.filter((point) => point.cluster_label === label);
    const color = points[0]?.cluster_color || "#31d4c8";

    return {
      type: "scatter",
      mode: "markers+text",
      name: label,
      x: points.map((point) => point.pc1),
      y: points.map((point) => point.pc2),
      ids: points.map((point) => point.team),
      text: points.map((point) => (point.model_rank <= 8 ? point.display_name : "")),
      textposition: "top center",
      textfont: { color: "#dcecff", size: 11 },
      marker: {
        size: points.map((point) => point.marker_size),
        color,
        opacity: 0.88,
        line: {
          color: "rgba(255,255,255,0.72)",
          width: 1,
        },
      },
      customdata: points.map((point) => [
        point.display_name,
        point.cluster_label,
        point.profile,
        point.confederation,
        point.score,
        point.model_rank,
        point.fifa_rank,
        point.elo,
        point.market,
        point.form,
        point.xg,
        point.xga,
        point.injuries,
        point.strengths,
      ]),
      hovertemplate:
        "<b>%{customdata[0]}</b><br>" +
        "Grupo: %{customdata[1]}<br>" +
        "Perfil: %{customdata[2]} · %{customdata[3]}<br>" +
        "Score Poder: %{customdata[4]:.3f}<br>" +
        "Ranking modelo: #%{customdata[5]}<br>" +
        "Ranking FIFA: %{customdata[6]}<br>" +
        "Elo: %{customdata[7]}<br>" +
        "Valor mercado: %{customdata[8]}<br>" +
        "Forma: %{customdata[9]}<br>" +
        "xG / xGA: %{customdata[10]} / %{customdata[11]}<br>" +
        "Fortalezas: %{customdata[13]}<extra></extra>",
    };
  });

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 18, r: 16, b: 58, l: 58 },
    xaxis: {
      title: { text: "Componente 1: Poder global", font: { color: "#dcecff" } },
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.16)",
      tickfont: { color: "#9cb4c9" },
    },
    yaxis: {
      title: { text: "Componente 2: Perfil competitivo", font: { color: "#dcecff" } },
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.16)",
      tickfont: { color: "#9cb4c9" },
    },
    legend: {
      orientation: "h",
      y: 1.12,
      x: 0,
      font: { color: "#dcecff" },
    },
    font: {
      family: "Inter, Segoe UI, sans-serif",
      color: "#eef6ff",
    },
    hoverlabel: {
      bgcolor: "#0d2138",
      bordercolor: "rgba(255,255,255,0.18)",
      font: { color: "#eef6ff" },
    },
  };

  Plotly.react("pca-chart", traces, layout, {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
  });

  const chart = document.getElementById("pca-chart");
  chart?.on("plotly_hover", (event) => {
    const team = event.points?.[0]?.id;
    renderPcaDetail(pointByTeam.get(team));
  });
  chart?.on("plotly_click", (event) => {
    const team = event.points?.[0]?.id;
    renderPcaDetail(pointByTeam.get(team));
  });

  const strongest = [...pcaMap.points].sort((a, b) => b.score - a.score)[0];
  renderPcaDetail(strongest);
}

function valueClass(metric, side) {
  if (side === "neutral") return "";
  return metric === side ? "is-best" : "is-worse";
}

function renderComparison(comparison) {
  comparisonBody.innerHTML = comparison
    .map(
      (item) => `
        <tr>
          <td class="metric-value ${valueClass("home", item.side)}">${item.home}</td>
          <td class="metric-name">${item.metric}</td>
          <td class="metric-value text-end ${valueClass("away", item.side)}">${item.away}</td>
        </tr>
      `,
    )
    .join("");
}

function updateMatchHeader(result) {
  document.getElementById("match-title").textContent = result.match.title;
  document.getElementById("match-flags").innerHTML = `
    <span>${result.match.home.flag}</span>
    <span>${result.match.away.flag}</span>
  `;

  document.getElementById("home-score-label").textContent = result.match.home.display_name;
  document.getElementById("away-score-label").textContent = result.match.away.display_name;
  document.getElementById("home-score").textContent = result.scores.home.toFixed(3);
  document.getElementById("away-score").textContent = result.scores.away.toFixed(3);
  document.getElementById("score-diff").textContent = result.scores.difference.toFixed(3);

  document.getElementById("comparison-home-head").textContent = result.match.home.display_name;
  document.getElementById("comparison-away-head").textContent = result.match.away.display_name;
  document.getElementById("comparison-title").textContent = result.match.title;
  document.getElementById("interpretation").innerHTML = `
    <i class="fa-solid fa-wand-magic-sparkles"></i>
    <span>${result.interpretation}</span>
  `;
}

async function runPrediction() {
  if (!teamASelect || !teamBSelect) return;

  const teamA = teamASelect.value;
  const teamB = teamBSelect.value;

  if (teamA === teamB) {
    showToast("Elige dos selecciones diferentes.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_a: teamA,
        team_b: teamB,
      }),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "No se pudo calcular el partido.");
    }

    updateMatchHeader(result);
    renderProbabilityBars(result.probabilities);
    renderChart(result.probabilities);
    renderComparison(result.comparison);
  } catch (error) {
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

window.addEventListener("resize", () => {
  if (window.Plotly) {
    ["probability-chart", "pca-chart"].forEach((id) => {
      const chart = document.getElementById(id);
      if (chart) {
        Plotly.Plots.resize(chart);
      }
    });
  }
});

if (bootstrapData.teams?.length) {
  window.addEventListener("DOMContentLoaded", () => {
    runPrediction();
    renderPcaMap(bootstrapData.pcaMap);
  });
}
