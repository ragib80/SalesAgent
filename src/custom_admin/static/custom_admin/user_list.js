/* Voice of Sales — user list (requires view or change permission) */
(function () {
  'use strict';

  const { $, esc, initials, fmtDate, fmtRelative, errorMessage, api, show, showDenied, showError } = window.VOS;

  const LEVELS = [
    { key: 'depo',      listKey: 'depos',       one: 'depo',      many: 'depos' },
    { key: 'territory', listKey: 'territories', one: 'territory', many: 'territories' },
    { key: 'zone',      listKey: 'zones',       one: 'zone',      many: 'zones' },
    { key: 'division',  listKey: 'divisions',   one: 'division',  many: 'divisions' },
  ];
  const ROLE_LABELS = { superuser: 'Superuser', staff: 'Staff', admin: 'Admin' };
  const FILTER_KEYS = ['q', 'status', 'provider', 'depo', 'territory', 'zone', 'division', 'missing_depo'];
  const DEFAULTS = { ordering: '-joined', page: '1', page_size: '20' };

  const els = {
    q: $('ua-q'), status: $('ua-f-status'), provider: $('ua-f-provider'),
    filtersToggle: $('ua-filters-toggle'), filters: $('ua-filters'), filtersCount: $('ua-filters-count'),
    missing: $('ua-f-missing'), filtersClear: $('ua-filters-clear'),
    wrap: $('ua-table-wrap'), rows: $('ua-rows'),
    range: $('ua-range'), pageLabel: $('ua-page-label'), prev: $('ua-prev'), next: $('ua-next'),
    pageSize: $('ua-page-size'), addBtn: $('ua-list-add'),
  };
  const levelSelect = key => $(`ua-f-${key}`);

  // State lives in the URL so reload / back keep the same view
  const state = Object.assign({}, DEFAULTS);
  const initial = new URLSearchParams(window.location.search);
  [...FILTER_KEYS, 'ordering', 'page', 'page_size'].forEach(k => {
    if (initial.get(k)) state[k] = initial.get(k);
  });

  let lastData = null;
  let requestSeq = 0;
  let canChange = false;

  function writeUrl() {
    const qs = new URLSearchParams();
    Object.entries(state).forEach(([k, v]) => { if (v && v !== DEFAULTS[k]) qs.set(k, v); });
    const url = `${window.location.pathname}${[...qs].length ? `?${qs}` : ''}`;
    window.history.replaceState(null, '', url);
  }

  function activeCoverageFilters() {
    return LEVELS.filter(l => state[l.key]).length + (state.missing_depo ? 1 : 0);
  }

  function anyFilter() {
    return FILTER_KEYS.some(k => state[k]);
  }

  /* ── Rendering ── */
  function coverageCell(u) {
    if (u.role) {
      return '<span class="ua-tag ua-tag--accent" title="Admin accounts see all sales data">Unrestricted</span>';
    }
    const c = u.coverage_counts;
    const tags = [];
    tags.push(c.depo
      ? `<span class="ua-tag">${c.depo} ${c.depo === 1 ? 'depo' : 'depos'}</span>`
      : '<span class="ua-tag ua-tag--warn" title="Can\'t run sales queries until a depo is assigned">No depo</span>');
    LEVELS.slice(1).forEach(l => {
      const n = c[l.key];
      if (n) tags.push(`<span class="ua-tag">${n} ${n === 1 ? l.one : l.many}</span>`);
    });
    return `<div class="ua-tags">${tags.join('')}</div>`;
  }

  function rowHtml(u) {
    const name = u.full_name || u.username;
    const href = `/api/manage/users/${encodeURIComponent(u.uuid)}/`;
    const role = u.role ? `<span class="ua-role">${ROLE_LABELS[u.role]}</span>` : '';
    return `
      <tr data-href="${href}">
        <td>
          <div class="ua-cell-user">
            <span class="ua-avatar ua-avatar--sm" aria-hidden="true">${esc(initials(u))}</span>
            <div class="ua-cell-user-text">
              <a class="ua-row-link" href="${href}">${esc(name)}</a>${role}
              <div class="ua-cell-sub">${esc(u.email || '—')}</div>
            </div>
          </div>
        </td>
        <td class="ua-col-md ua-mono">${esc(u.username)}</td>
        <td>${coverageCell(u)}</td>
        <td class="ua-col-lg ua-cell-muted">${esc(u.identity_provider_label)}</td>
        <td>${u.is_active
          ? '<span class="ua-status ua-status--on">Active</span>'
          : '<span class="ua-status">Inactive</span>'}</td>
        <td class="ua-col-md ua-cell-muted" title="${esc(fmtDate(u.last_login, true))}">${esc(fmtRelative(u.last_login))}</td>
        <td class="ua-col-lg ua-cell-muted">${esc(fmtDate(u.date_joined))}</td>
        <td class="ua-cell-action">
          <a class="ua-icon-btn" href="${href}" tabindex="-1" aria-hidden="true">
            <i class="bi ${canChange ? 'bi-pencil' : 'bi-chevron-right'}"></i>
          </a>
        </td>
      </tr>`;
  }

  function skeletonRows(n) {
    return Array.from({ length: n }, () => `
      <tr class="ua-skeleton-row">
        <td><div class="ua-cell-user"><span class="ua-avatar ua-avatar--sm ua-skel"></span><span class="ua-skel ua-skel--line"></span></div></td>
        <td class="ua-col-md"><span class="ua-skel ua-skel--short"></span></td>
        <td><span class="ua-skel ua-skel--short"></span></td>
        <td class="ua-col-lg"><span class="ua-skel ua-skel--short"></span></td>
        <td><span class="ua-skel ua-skel--tiny"></span></td>
        <td class="ua-col-md"><span class="ua-skel ua-skel--short"></span></td>
        <td class="ua-col-lg"><span class="ua-skel ua-skel--short"></span></td>
        <td></td>
      </tr>`).join('');
  }

  function renderRows(data) {
    if (!data.results.length) {
      els.rows.innerHTML = `
        <tr><td colspan="8">
          <div class="ua-empty">
            ${anyFilter()
              ? 'No users match these filters. <button type="button" class="ua-link" data-action="clear">Clear filters</button>'
              : 'No users yet.'}
          </div>
        </td></tr>`;
      return;
    }
    els.rows.innerHTML = data.results.map(rowHtml).join('');
  }

  function renderPager(data) {
    const start = data.count ? (data.page - 1) * data.page_size + 1 : 0;
    const end = Math.min(data.page * data.page_size, data.count);
    els.range.textContent = data.count
      ? `${start}–${end} of ${data.count} ${data.count === 1 ? 'user' : 'users'}`
      : '0 users';
    els.pageLabel.textContent = `Page ${data.page} of ${data.num_pages}`;
    els.prev.disabled = data.page <= 1;
    els.next.disabled = data.page >= data.num_pages;
  }

  function renderSortHeaders() {
    const key = state.ordering.replace(/^-/, '');
    const desc = state.ordering.startsWith('-');
    document.querySelectorAll('.ua-sort').forEach(btn => {
      const on = btn.dataset.sort === key;
      const th = btn.closest('th');
      btn.classList.toggle('is-sorted', on);
      btn.dataset.dir = on ? (desc ? 'desc' : 'asc') : '';
      if (on) th.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
      else th.removeAttribute('aria-sort');
    });
  }

  function syncControls() {
    els.q.value = state.q || '';
    els.status.value = state.status || '';
    els.provider.value = state.provider || '';
    LEVELS.forEach(l => { levelSelect(l.key).value = state[l.key] || ''; });
    els.missing.checked = state.missing_depo === '1';
    els.pageSize.value = state.page_size;
    const n = activeCoverageFilters();
    els.filtersCount.hidden = !n;
    els.filtersCount.textContent = n;
    els.filtersCount.classList.toggle('has-value', !!n);
    if (n) setFiltersOpen(true);
  }

  function setFiltersOpen(open) {
    els.filters.hidden = !open;
    els.filtersToggle.setAttribute('aria-expanded', open);
  }

  /* ── Loading ── */
  async function load() {
    const seq = ++requestSeq;
    writeUrl();
    renderSortHeaders();
    els.wrap.setAttribute('aria-busy', 'true');
    els.wrap.classList.add('is-loading');
    if (!lastData) els.rows.innerHTML = skeletonRows(6);

    const params = {};
    [...FILTER_KEYS, 'ordering', 'page', 'page_size'].forEach(k => { params[k] = state[k] || ''; });

    try {
      const res = await api('/manage/api/users/', { params });
      if (seq !== requestSeq) return;  // a newer request is in flight
      if (res.status === 403) {
        showDenied("You can't view users", errorMessage(res.data, 'Ask an administrator for permission.'));
        return;
      }
      if (!res.ok) {
        els.rows.innerHTML = `<tr><td colspan="8"><div class="ua-empty ua-empty--error">${
          esc(errorMessage(res.data, "Couldn't load users. Try again."))}</div></td></tr>`;
        return;
      }
      lastData = res.data;
      state.page = String(res.data.page);
      writeUrl();
      renderRows(res.data);
      renderPager(res.data);
    } catch (_) {
      if (seq === requestSeq) {
        els.rows.innerHTML = '<tr><td colspan="8"><div class="ua-empty ua-empty--error">Couldn\'t reach the server. Check your connection and try again.</div></td></tr>';
      }
    } finally {
      if (seq === requestSeq) {
        els.wrap.setAttribute('aria-busy', 'false');
        els.wrap.classList.remove('is-loading');
      }
    }
  }

  function setFilter(key, value) {
    state[key] = value;
    state.page = '1';
    syncControls();
    load();
  }

  function clearFilters() {
    FILTER_KEYS.forEach(k => { state[k] = ''; });
    state.page = '1';
    syncControls();
    load();
  }

  /* ── Events ── */
  let searchTimer = null;
  els.q.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => setFilter('q', els.q.value.trim()), 300);
  });
  els.status.addEventListener('change', () => setFilter('status', els.status.value));
  els.provider.addEventListener('change', () => setFilter('provider', els.provider.value));
  LEVELS.forEach(l => levelSelect(l.key).addEventListener('change', e => setFilter(l.key, e.target.value)));
  els.missing.addEventListener('change', () => setFilter('missing_depo', els.missing.checked ? '1' : ''));
  els.filtersClear.addEventListener('click', clearFilters);
  els.filtersToggle.addEventListener('click', () => setFiltersOpen(els.filters.hidden));

  els.pageSize.addEventListener('change', () => {
    state.page_size = els.pageSize.value;
    state.page = '1';
    load();
  });
  els.prev.addEventListener('click', () => { state.page = String(Math.max(1, Number(state.page) - 1)); load(); });
  els.next.addEventListener('click', () => { state.page = String(Number(state.page) + 1); load(); });

  document.querySelectorAll('.ua-sort').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.sort;
      const current = state.ordering.replace(/^-/, '');
      // Dates default to newest first, text to A–Z
      const firstDesc = key === 'joined' || key === 'last_login';
      if (current === key) state.ordering = state.ordering.startsWith('-') ? key : `-${key}`;
      else state.ordering = firstDesc ? `-${key}` : key;
      state.page = '1';
      load();
    });
  });

  els.rows.addEventListener('click', e => {
    if (e.target.closest('[data-action="clear"]')) { clearFilters(); return; }
    if (e.target.closest('a, button')) return;
    const tr = e.target.closest('tr[data-href]');
    if (tr) window.location.href = tr.dataset.href;
  });

  /* ── Init ── */
  async function init() {
    let perms;
    try {
      perms = await VOS.loadPerms();
    } catch (_) {
      showError();
      return;
    }
    if (!perms.view) {
      showDenied("You can't view users",
        'Viewing users needs the Can view SalesAuthUser or Can change SalesAuthUser permission. Ask an administrator to grant it.');
      return;
    }
    canChange = perms.change;
    els.addBtn.hidden = !perms.add;

    // Coverage options feed the filter dropdowns; the list still works without them
    api('/manage/api/coverage-options/').then(res => {
      if (!res.ok) return;
      LEVELS.forEach(l => {
        const sel = levelSelect(l.key);
        sel.insertAdjacentHTML('beforeend', (res.data[l.listKey] || []).map(o =>
          `<option value="${o.id}">${esc(o.code)}${o.name ? ` — ${esc(o.name)}` : ''}</option>`).join(''));
        sel.value = state[l.key] || '';
      });
    }).catch(() => {});

    syncControls();
    show('ua-content');
    load();
  }

  init();
})();
