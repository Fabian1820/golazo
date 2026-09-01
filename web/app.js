'use strict';

const $ = (id) => document.getElementById(id);
const pct = (v) => (v * 100).toFixed(1) + '%';

const els = {
  league: $('league'), home: $('home'), away: $('away'), kickoff: $('kickoff'),
  form: $('fixture-form'), submit: $('submit'), error: $('error'),
  results: $('results'), warnings: $('warnings'),
  bars: $('bars'), xg: $('xg'), ou: $('ou'), scores: $('scores'), btts: $('btts'),
};

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || body.error || `Error ${res.status}`);
  return body;
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
}

function fill(select, items, placeholder) {
  select.innerHTML = '';
  const opt = document.createElement('option');
  opt.value = '';
  opt.textContent = placeholder;
  opt.disabled = true;
  opt.selected = true;
  select.appendChild(opt);
  for (const item of items) {
    const o = document.createElement('option');
    o.value = item;
    o.textContent = item;
    select.appendChild(o);
  }
}

// --- carga inicial ---------------------------------------------------------

async function init() {
  try {
    const model = await api('/api/model');
    $('model-version').textContent = model.version;
    $('model-trained').textContent = `datos hasta ${model.train_end}`;
    $('model-badge').hidden = false;
    const bt = model.backtest || {};
    $('foot-model').textContent = bt.rps
      ? `${model.name} · RPS ${bt.rps.toFixed(4)} sobre ${bt.n_matches} partidos de backtest`
      : `${model.name} · ${model.n_matches} partidos de entrenamiento`;

    // El texto explicativo cita el tamaño del backtest: se toma del modelo
    // servido para que no pueda quedarse obsoleto al reentrenar.
    if (bt.n_matches) {
      $('n-backtest').textContent = bt.n_matches.toLocaleString('es-ES');
    }

    const leagues = await api('/api/leagues');
    fill(els.league, leagues, 'Elegir liga…');
  } catch (err) {
    showError(`No se pudo cargar el modelo: ${err.message}`);
  }
}

els.league.addEventListener('change', async () => {
  clearError();
  els.home.disabled = els.away.disabled = true;
  els.submit.disabled = true;
  try {
    const teams = await api(`/api/teams/${encodeURIComponent(els.league.value)}`);
    fill(els.home, teams, 'Equipo local…');
    fill(els.away, teams, 'Equipo visitante…');
    els.home.disabled = els.away.disabled = false;
  } catch (err) {
    showError(err.message);
  }
});

const checkReady = () => {
  els.submit.disabled = !(els.home.value && els.away.value);
};
els.home.addEventListener('change', checkReady);
els.away.addEventListener('change', checkReady);

// --- predicción ------------------------------------------------------------

els.form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  clearError();

  if (els.home.value === els.away.value) {
    showError('Un equipo no puede jugar contra sí mismo.');
    return;
  }

  els.submit.disabled = true;
  els.submit.textContent = 'Calculando…';

  try {
    const body = {
      home: els.home.value,
      away: els.away.value,
      league: els.league.value,
    };
    if (els.kickoff.value) body.kickoff = `${els.kickoff.value}T15:00:00`;

    render(await api('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }));
  } catch (err) {
    showError(err.message);
  } finally {
    els.submit.disabled = false;
    els.submit.textContent = 'Calcular probabilidades';
  }
});

// --- pintado ---------------------------------------------------------------

function render(p) {
  $('r-home').textContent = p.home;
  $('r-away').textContent = p.away;
  $('p-version').textContent = p.model_version;
  $('p-trained').textContent = p.trained_through;

  els.warnings.innerHTML = '';
  for (const w of p.warnings || []) {
    const div = document.createElement('div');
    div.className = 'warning';
    div.textContent = w;
    els.warnings.appendChild(div);
  }

  const rows = [
    ['Gana ' + p.home, p.probabilities.H, 'var(--home)'],
    ['Empate', p.probabilities.D, 'var(--draw)'],
    ['Gana ' + p.away, p.probabilities.A, 'var(--away)'],
  ];
  els.bars.innerHTML = '';
  for (const [name, value, color] of rows) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="name"></span>
      <div class="bar-track"><div class="bar-fill" style="width:0;background:${color}"></div></div>
      <span class="pct">${pct(value)}</span>`;
    row.querySelector('.name').textContent = name;
    els.bars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector('.bar-fill').style.width = `${value * 100}%`;
    });
  }

  const m = p.markets;
  els.results.hidden = false;
  if (!m) return;

  const eg = m.expected_goals;
  els.xg.innerHTML = `
    <div><div class="val">${eg.home.toFixed(2)}</div><div class="lbl">local</div></div>
    <div><div class="val">${eg.away.toFixed(2)}</div><div class="lbl">visitante</div></div>
    <div><div class="val">${eg.total.toFixed(2)}</div><div class="lbl">total</div></div>`;

  els.ou.innerHTML = '<tr><th>línea</th><th>más de</th><th>menos de</th></tr>' +
    Object.entries(m.over_under).map(([line, v]) =>
      `<tr><td>${line}</td><td>${pct(v.over)}</td><td>${pct(v.under)}</td></tr>`).join('');

  const top = m.top_scorelines[0].probability;
  els.scores.innerHTML = m.top_scorelines.map((s) => `
    <li>
      <span class="sc">${s.score}</span>
      <span class="tr"><span class="fl" style="width:${(s.probability / top) * 100}%"></span></span>
      <span class="pr">${pct(s.probability)}</span>
    </li>`).join('');

  els.btts.innerHTML = `
    <div><div class="v">${pct(m.both_teams_score.yes)}</div><div class="l">sí</div></div>
    <div><div class="v">${pct(m.both_teams_score.no)}</div><div class="l">no</div></div>`;
}

init();
