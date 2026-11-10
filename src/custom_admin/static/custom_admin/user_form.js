/* Voice of Sales — add / edit user form
   add  : requires add permission
   edit : requires view (read-only) or change permission; the server decides can_edit per user */
(function () {
  'use strict';

  const { $, esc, initials, fmtDate, fmtRelative, errorMessage, api, show, showDenied, showError, toast } = window.VOS;

  const LEVELS = [
    { key: 'depo',      label: 'Depo',      plural: 'depos',       listKey: 'depos',       payloadKey: 'depo_ids', required: true },
    { key: 'territory', label: 'Territory', plural: 'territories', listKey: 'territories', payloadKey: 'territory_ids' },
    { key: 'zone',      label: 'Zone',      plural: 'zones',       listKey: 'zones',       payloadKey: 'zone_ids' },
    { key: 'division',  label: 'Division',  plural: 'divisions',   listKey: 'divisions',   payloadKey: 'division_ids' },
  ];
  const ROLE_LABELS = { superuser: 'Superuser', staff: 'Staff', admin: 'Admin' };
  const CHIP_LIMIT = 6;

  const content = $('ua-content');
  const MODE = content.dataset.mode;          // 'add' | 'edit'
  const USER_UUID = content.dataset.userUuid;

  const state = {
    options: {},          // level key -> [{id, code, name}]
    byId: {},             // level key -> Map(id -> option)
    selected: {},         // level key -> Set(id)
    active: 'depo',
    view: 'all',
    search: '',
    readonly: false,
    // add
    profile: null,
    lookedUpIdentifier: '',
    // edit
    user: null,
    baseline: '',
  };
  LEVELS.forEach(l => { state.selected[l.key] = new Set(); state.byId[l.key] = new Map(); });

  const els = {
    form: $('ua-form'),
    tabs: $('ua-tabs'), panel: $('ua-panel'), search: $('ua-search'), options: $('ua-options'),
    pickerWrap: $('ua-picker-wrap'), covReadonly: $('ua-cov-readonly'), covHelp: $('ua-cov-help'),
    shownCount: $('ua-shown-count'), selectShown: $('ua-select-shown'), clear: $('ua-clear'),
    passAvatar: $('ua-pass-avatar'), passName: $('ua-pass-name'), passSub: $('ua-pass-sub'),
    passLevels: $('ua-pass-levels'), passNote: $('ua-pass-note'),
    submit: $('ua-submit'), submitHint: $('ua-submit-hint'), submitMsg: $('ua-submit-msg'),
    noDepoDialog: $('ua-nodepo-dialog'),
    // add
    identifier: $('ua-identifier'), lookupBtn: $('ua-lookup-btn'), lookupMsg: $('ua-lookup-msg'),
    identity: $('ua-identity'), identityAvatar: $('ua-identity-avatar'),
    identityName: $('ua-identity-name'), identityMeta: $('ua-identity-meta'),
    // edit
    crumb: $('ua-crumb'), editTitle: $('ua-edit-title'), editSub: $('ua-edit-sub'),
    banner: $('ua-readonly-banner'), bannerText: $('ua-readonly-text'),
    syncBtn: $('ua-sync-btn'), firstName: $('ua-first-name'), lastName: $('ua-last-name'), activeToggle: $('ua-active'),
    discard: $('ua-discard'),
  };

  /* ── Helpers ── */
  function optionLabel(opt) {
    return opt.code && opt.name ? `${opt.code} — ${opt.name}` : (opt.name || opt.code || String(opt.id));
  }

  function setBusy(btn, busy) {
    btn.classList.toggle('is-busy', busy);
    btn.disabled = busy;
  }

  function setOptions(data) {
    LEVELS.forEach(l => {
      state.options[l.key] = data[l.listKey] || [];
      state.byId[l.key] = new Map(state.options[l.key].map(o => [o.id, o]));
    });
  }

  /* ── Coverage picker ── */
  function renderTabs() {
    els.tabs.innerHTML = LEVELS.map(l => {
      const count = state.selected[l.key].size;
      const active = l.key === state.active;
      return `
        <button type="button" class="ua-tab" role="tab" id="ua-tab-${l.key}" data-level="${l.key}"
                aria-selected="${active}" aria-controls="ua-panel" tabindex="${active ? 0 : -1}">
          ${l.label}${l.required ? '<span class="ua-tab-req" aria-label="required">*</span>' : ''}
          <span class="ua-tab-count ${count ? 'has-value' : ''}">${count}</span>
        </button>`;
    }).join('');
    els.panel.setAttribute('aria-labelledby', `ua-tab-${state.active}`);
  }

  function visibleOptions() {
    const sel = state.selected[state.active];
    const q = state.search.toLowerCase();
    return (state.options[state.active] || []).filter(o => {
      if (state.view === 'selected' && !sel.has(o.id)) return false;
      if (!q) return true;
      return String(o.code).toLowerCase().includes(q) || String(o.name).toLowerCase().includes(q);
    });
  }

  function updatePickerActions() {
    const sel = state.selected[state.active];
    const list = visibleOptions();
    els.selectShown.disabled = !list.length || list.every(o => sel.has(o.id));
    els.clear.disabled = !sel.size;
  }

  function renderOptions() {
    const level = LEVELS.find(l => l.key === state.active);
    const sel = state.selected[state.active];
    const list = visibleOptions();
    const total = (state.options[state.active] || []).length;

    els.shownCount.textContent = list.length === total
      ? `${total} ${level.plural}`
      : `Showing ${list.length} of ${total}`;
    updatePickerActions();

    if (!list.length) {
      let msg = `No ${level.plural} match “${esc(state.search)}”.`;
      if (state.view === 'selected' && !state.search) msg = `No ${level.plural} selected yet.`;
      if (!total) msg = `No ${level.plural} available.`;
      els.options.innerHTML = `<div class="ua-empty">${msg}</div>`;
      return;
    }

    els.options.innerHTML = list.map(o => `
      <label class="ua-option" title="${esc(optionLabel(o))}">
        <input type="checkbox" value="${o.id}" ${sel.has(o.id) ? 'checked' : ''}>
        <span class="ua-check" aria-hidden="true"><i class="bi bi-check-lg"></i></span>
        <span class="ua-option-text">
          <span class="ua-option-code">${esc(o.code || o.name)}</span>
          ${o.code && o.name ? `<span class="ua-option-name">${esc(o.name)}</span>` : ''}
        </span>
      </label>`).join('');
  }

  function renderReadonlyCoverage() {
    const unrestricted = !!(state.user && state.user.role);
    els.covReadonly.innerHTML = LEVELS.map(l => {
      const items = [...state.selected[l.key]].map(id => state.byId[l.key].get(id)).filter(Boolean);
      const body = items.length
        ? items.map(o => `<span class="ua-chip ua-chip--static" title="${esc(optionLabel(o))}"><span>${esc(optionLabel(o))}</span></span>`).join('')
        : `<span class="ua-cell-muted">${l.required && !unrestricted ? 'None assigned' : `All ${l.plural}`}</span>`;
      return `<div class="ua-level"><dt>${l.label}<span class="ua-level-state">${items.length || ''}</span></dt><dd>${body}</dd></div>`;
    }).join('');
  }

  /* ── Summary ── */
  function summaryPerson() {
    if (MODE === 'add') return state.profile;
    if (!state.user) return null;
    const first = els.firstName.value.trim();
    const last = els.lastName.value.trim();
    return { ...state.user, first_name: first, last_name: last, full_name: `${first} ${last}`.trim() };
  }

  function renderSummary() {
    const p = summaryPerson();
    if (p) {
      els.passAvatar.className = 'ua-avatar';
      els.passAvatar.textContent = initials(p);
      els.passName.textContent = p.full_name || p.username;
      els.passSub.textContent = p.email || p.username;
    } else {
      els.passAvatar.className = 'ua-avatar ua-avatar--empty';
      els.passAvatar.innerHTML = '<i class="bi bi-person"></i>';
      els.passName.textContent = 'No one selected yet';
      els.passSub.textContent = 'Find a person in step 1';
    }

    const unrestricted = MODE === 'edit' && state.user && state.user.role;
    els.passLevels.innerHTML = LEVELS.map(l => {
      const ids = [...state.selected[l.key]];
      let stateText, stateClass = '';
      if (ids.length) {
        stateText = `${ids.length} ${ids.length === 1 ? l.label.toLowerCase() : l.plural}`;
      } else if (l.required && !unrestricted) {
        stateText = 'None selected';
        stateClass = 'is-missing';
      } else {
        stateText = `All ${l.plural}`;
        stateClass = 'is-all';
      }
      const chips = ids.slice(0, CHIP_LIMIT).map(id => {
        const opt = state.byId[l.key].get(id);
        const label = opt ? optionLabel(opt) : String(id);
        const remove = state.readonly ? '' :
          `<button type="button" class="ua-chip-x" data-level="${l.key}" data-id="${id}" aria-label="Remove ${esc(label)}">×</button>`;
        return `<span class="ua-chip ${state.readonly ? 'ua-chip--static' : ''}" title="${esc(label)}">` +
               `<span>${esc(opt ? (opt.code || opt.name) : id)}</span>${remove}</span>`;
      }).join('');
      const more = ids.length > CHIP_LIMIT
        ? `<span class="ua-chip ua-chip--more">+${ids.length - CHIP_LIMIT} more</span>` : '';
      return `
        <div class="ua-level">
          <dt>${l.label}<span class="ua-level-state ${stateClass}">${stateText}</span></dt>
          <dd>${chips}${more}</dd>
        </div>`;
    }).join('');

    els.passNote.hidden = !!unrestricted || state.selected.depo.size > 0;
    if (unrestricted) {
      els.passNote.hidden = false;
      els.passNote.classList.add('ua-pass-note--info');
      els.passNote.querySelector('span').textContent =
        'This is an administrator account. It can see all sales data regardless of coverage.';
    }

    if (MODE === 'add') {
      els.submit.disabled = !p;
      els.submitHint.textContent = p ? '' : 'Find the person first.';
    } else if (!state.readonly) {
      const dirty = isDirty();
      els.submit.disabled = !dirty;
      els.discard.hidden = !dirty;
      els.submitHint.textContent = dirty ? 'You have unsaved changes.' : 'No changes yet.';
    }
  }

  function renderAll() {
    renderTabs();
    renderOptions();
    renderSummary();
  }

  function switchTab(key, focus) {
    state.active = key;
    state.search = '';
    state.view = 'all';
    els.search.value = '';
    els.panel.querySelectorAll('.ua-seg-btn').forEach(b => {
      const on = b.dataset.view === 'all';
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', on);
    });
    renderTabs();
    renderOptions();
    els.options.scrollTop = 0;
    if (focus) $(`ua-tab-${key}`).focus();
  }

  /* ── Picker events ── */
  els.tabs.addEventListener('click', e => {
    const tab = e.target.closest('.ua-tab');
    if (tab) switchTab(tab.dataset.level);
  });
  els.tabs.addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const i = LEVELS.findIndex(l => l.key === state.active);
    const next = LEVELS[(i + (e.key === 'ArrowRight' ? 1 : LEVELS.length - 1)) % LEVELS.length];
    e.preventDefault();
    switchTab(next.key, true);
  });
  els.search.addEventListener('input', () => {
    state.search = els.search.value.trim();
    renderOptions();
  });
  els.panel.querySelectorAll('.ua-seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      els.panel.querySelectorAll('.ua-seg-btn').forEach(b => {
        const on = b === btn;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-pressed', on);
      });
      renderOptions();
    });
  });
  els.options.addEventListener('change', e => {
    if (e.target.type !== 'checkbox') return;
    const id = Number(e.target.value);
    const sel = state.selected[state.active];
    if (e.target.checked) sel.add(id); else sel.delete(id);
    renderTabs();
    renderSummary();
    updatePickerActions();
  });
  els.selectShown.addEventListener('click', () => {
    const sel = state.selected[state.active];
    visibleOptions().forEach(o => sel.add(o.id));
    renderAll();
  });
  els.clear.addEventListener('click', () => {
    state.selected[state.active].clear();
    renderAll();
  });
  els.passLevels.addEventListener('click', e => {
    const x = e.target.closest('.ua-chip-x');
    if (!x) return;
    state.selected[x.dataset.level].delete(Number(x.dataset.id));
    renderAll();
  });

  /* ── No-depo confirmation ── */
  function confirmWithoutDepo() {
    const dlg = els.noDepoDialog;
    if (typeof dlg.showModal !== 'function') {
      return Promise.resolve(window.confirm("Continue without a depo? This user won't be able to run sales queries."));
    }
    return new Promise(resolve => {
      const onClick = e => {
        const btn = e.target.closest('[data-close]');
        if (btn) dlg.close(btn.dataset.close);
      };
      dlg.addEventListener('click', onClick);
      dlg.addEventListener('close', () => {
        dlg.removeEventListener('click', onClick);
        resolve(dlg.returnValue === 'confirm');
      }, { once: true });
      dlg.returnValue = '';
      dlg.showModal();
    });
  }

  function coveragePayload() {
    const body = {};
    LEVELS.forEach(l => { body[l.payloadKey] = [...state.selected[l.key]]; });
    return body;
  }

  function describeCoverage() {
    return LEVELS.map(l => {
      const n = state.selected[l.key].size;
      if (n) return `${n} ${n === 1 ? l.label.toLowerCase() : l.plural}`;
      return l.required ? `no ${l.label.toLowerCase()}` : `all ${l.plural}`;
    }).join(', ');
  }

  /* ================================================================ ADD */
  function initAddMode(perms) {
    function setLookupMessage(text, isError) {
      els.lookupMsg.textContent = text || '';
      els.lookupMsg.classList.toggle('is-error', !!isError);
      els.identifier.setAttribute('aria-invalid', isError ? 'true' : 'false');
    }

    function setProfile(profile) {
      state.profile = profile;
      els.identity.hidden = !profile;
      if (profile) {
        els.identityAvatar.textContent = initials(profile);
        els.identityName.textContent = profile.full_name || profile.username;
        els.identityMeta.textContent = profile.email ? `${profile.email} (${profile.username})` : profile.username;
      }
      renderSummary();
    }

    async function lookup() {
      const identifier = els.identifier.value.trim();
      if (!identifier) {
        setLookupMessage('Enter a work email or Microsoft username.', true);
        els.identifier.focus();
        return;
      }
      setBusy(els.lookupBtn, true);
      setLookupMessage('Searching the Microsoft directory…');
      setProfile(null);
      try {
        const { ok, data } = await api('/manage/api/users/lookup/', { method: 'POST', body: { identifier } });
        if (ok) {
          state.lookedUpIdentifier = identifier;
          setLookupMessage('');
          setProfile(data.profile);
        } else {
          setLookupMessage(errorMessage(data, "Couldn't look up this user. Try again."), true);
        }
      } catch (_) {
        setLookupMessage("Couldn't reach the server. Check your connection and try again.", true);
      } finally {
        setBusy(els.lookupBtn, false);
      }
    }

    els.lookupBtn.addEventListener('click', lookup);
    els.identifier.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); lookup(); }
    });
    els.identifier.addEventListener('input', () => {
      // Editing the identifier invalidates a previous lookup
      if (state.profile && els.identifier.value.trim() !== state.lookedUpIdentifier) {
        setProfile(null);
        setLookupMessage('Find the user again to confirm the change.');
      } else if (els.lookupMsg.classList.contains('is-error')) {
        setLookupMessage('');
      }
    });

    els.form.addEventListener('submit', async e => {
      e.preventDefault();
      if (!state.profile) return;
      els.submitMsg.textContent = '';

      if (!state.selected.depo.size && !(await confirmWithoutDepo())) {
        switchTab('depo', true);
        return;
      }

      setBusy(els.submit, true);
      try {
        const { ok, data } = await api('/manage/api/users/', {
          method: 'POST',
          body: { identifier: state.lookedUpIdentifier, ...coveragePayload() },
        });
        if (ok) {
          const name = data.full_name || data.username;
          $('ua-success-title').textContent = `${name} added`;
          $('ua-success-text').textContent =
            `They can now sign in to Voice of Sales with their Microsoft account (${data.email || data.username}). ` +
            `Access: ${describeCoverage()}.`;
          const open = $('ua-success-open');
          open.hidden = !perms.view;
          open.href = `/api/manage/users/${encodeURIComponent(data.uuid)}/`;
          $('ua-success-list').hidden = !perms.view;
          show('ua-success');
        } else {
          els.submitMsg.textContent = errorMessage(data, "Couldn't add this user. Try again.");
        }
      } catch (_) {
        els.submitMsg.textContent = "Couldn't reach the server. Check your connection and try again.";
      } finally {
        setBusy(els.submit, false);
        els.submit.disabled = !state.profile;
      }
    });

    $('ua-add-another').addEventListener('click', () => {
      LEVELS.forEach(l => state.selected[l.key].clear());
      state.lookedUpIdentifier = '';
      els.identifier.value = '';
      els.submitMsg.textContent = '';
      setLookupMessage('');
      setProfile(null);
      switchTab('depo');
      renderAll();
      show('ua-content');
      els.identifier.focus();
    });
  }

  /* ================================================================ EDIT */
  function snapshot() {
    const cov = {};
    LEVELS.forEach(l => { cov[l.key] = [...state.selected[l.key]].sort((a, b) => a - b); });
    return JSON.stringify({
      first: els.firstName.value.trim(),
      last: els.lastName.value.trim(),
      active: els.activeToggle.checked,
      cov,
    });
  }

  function isDirty() {
    return MODE === 'edit' && !state.readonly && !!state.user && snapshot() !== state.baseline;
  }

  function applyUser(user) {
    state.user = user;
    const name = user.full_name || user.username;
    document.title = `${name} · Voice of Sales`;
    els.editTitle.textContent = name;
    els.editSub.innerHTML =
      `${user.is_active ? '<span class="ua-status ua-status--on">Active</span>' : '<span class="ua-status">Inactive</span>'}` +
      `${user.role ? `<span class="ua-role">${ROLE_LABELS[user.role]}</span>` : ''}`;

    $('ua-f-username').textContent = user.username;
    $('ua-f-email').textContent = user.email || '—';
    $('ua-f-provider').textContent = user.identity_provider_label +
      (user.is_microsoft_linked ? '' : user.identity_provider === 'microsoft' ? ' (not linked yet)' : '');
    $('ua-f-joined').textContent = fmtDate(user.date_joined) || '—';
    const ll = $('ua-f-last-login');
    ll.textContent = fmtRelative(user.last_login);
    ll.title = fmtDate(user.last_login, true);
    $('ua-f-role').textContent = user.role ? `${ROLE_LABELS[user.role]} (sees all sales data)` : 'Standard user';

    els.firstName.value = user.first_name || '';
    els.lastName.value = user.last_name || '';
    els.activeToggle.checked = !!user.is_active;

    LEVELS.forEach(l => {
      state.selected[l.key] = new Set((user.coverage[l.key] || []).map(o => o.id));
      // Make sure assigned items always have a label, even if missing from the master list
      (user.coverage[l.key] || []).forEach(o => { if (!state.byId[l.key].has(o.id)) state.byId[l.key].set(o.id, o); });
    });
    state.baseline = snapshot();
  }

  function setReadonly(reason) {
    state.readonly = true;
    els.banner.hidden = false;
    els.bannerText.textContent = reason;
    document.querySelectorAll('[data-editable]').forEach(el => { el.disabled = true; });
    els.pickerWrap.hidden = true;
    els.covReadonly.hidden = false;
    els.covHelp.textContent = 'The sales data this user can see. Empty levels allow everything.';
    els.submit.hidden = true;
    els.submitHint.textContent = '';
    renderReadonlyCoverage();
  }

  async function initEditMode(perms) {
    els.crumb.hidden = !perms.view;

    const res = await api(`/manage/api/users/${encodeURIComponent(USER_UUID)}/`);
    if (res.status === 404) { showError('This user may have been removed.', 'User not found'); return false; }
    if (res.status === 403) { showDenied("You can't view users", errorMessage(res.data, 'Ask an administrator for permission.')); return false; }
    if (!res.ok) { showError(errorMessage(res.data, 'Check your connection and try again.')); return false; }

    const user = res.data;
    if (user.can_edit) {
      const opts = await api('/manage/api/coverage-options/');
      if (!opts.ok) { showError(errorMessage(opts.data, 'Check your connection and try again.'), "Couldn't load coverage lists"); return false; }
      setOptions(opts.data);
    }
    applyUser(user);
    if (!user.can_edit) setReadonly(user.edit_block_reason || "You can't change this user.");

    [els.firstName, els.lastName].forEach(inp => inp.addEventListener('input', renderSummary));
    els.activeToggle.addEventListener('change', renderSummary);

    els.discard.addEventListener('click', () => {
      applyUser(state.user);
      els.submitMsg.textContent = '';
      renderAll();
    });

    window.addEventListener('beforeunload', e => {
      if (isDirty()) { e.preventDefault(); e.returnValue = ''; }
    });

    els.syncBtn.addEventListener('click', async () => {
      if (state.readonly) return;
      setBusy(els.syncBtn, true);
      try {
        const { ok, data } = await api(`/manage/api/users/${encodeURIComponent(USER_UUID)}/sync/`, { method: 'POST' });
        if (!ok) { toast(errorMessage(data, "Couldn't refresh from Microsoft."), 'error'); return; }
        // Keep unsaved coverage / status edits; take name + email from the directory
        const pending = { cov: LEVELS.map(l => new Set(state.selected[l.key])), active: els.activeToggle.checked };
        applyUser(data.user);
        LEVELS.forEach((l, i) => { state.selected[l.key] = pending.cov[i]; });
        els.activeToggle.checked = pending.active;
        renderAll();
        toast(data.changed.length
          ? `Updated from Microsoft: ${data.changed.map(f => f.replace('_', ' ')).join(', ')}`
          : 'Already up to date with Microsoft');
      } catch (_) {
        toast("Couldn't reach the server. Check your connection and try again.", 'error');
      } finally {
        setBusy(els.syncBtn, false);
      }
    });

    els.form.addEventListener('submit', async e => {
      e.preventDefault();
      if (state.readonly || !isDirty()) return;
      els.submitMsg.textContent = '';

      if (!state.user.role && !state.selected.depo.size && !(await confirmWithoutDepo())) {
        switchTab('depo', true);
        return;
      }

      setBusy(els.submit, true);
      try {
        const { ok, data } = await api(`/manage/api/users/${encodeURIComponent(USER_UUID)}/`, {
          method: 'PATCH',
          body: {
            first_name: els.firstName.value.trim(),
            last_name: els.lastName.value.trim(),
            is_active: els.activeToggle.checked,
            ...coveragePayload(),
          },
        });
        if (ok) {
          applyUser(data);
          renderAll();
          toast('Changes saved');
        } else {
          els.submitMsg.textContent = errorMessage(data, "Couldn't save changes. Try again.");
        }
      } catch (_) {
        els.submitMsg.textContent = "Couldn't reach the server. Check your connection and try again.";
      } finally {
        setBusy(els.submit, false);
        renderSummary();
      }
    });
    return true;
  }

  /* ── Init ── */
  async function init() {
    let perms;
    try {
      perms = await VOS.loadPerms();
    } catch (_) {
      showError();
      return;
    }

    try {
      if (MODE === 'add') {
        if (!perms.add) {
          showDenied("You can't add users",
            'Adding users needs the Can add SalesAuthUser permission. Ask an administrator to grant it.');
          return;
        }
        const res = await api('/manage/api/coverage-options/');
        if (!res.ok) {
          showError(errorMessage(res.data, 'Check your connection and try again.'), "Couldn't load coverage lists");
          return;
        }
        setOptions(res.data);
        initAddMode(perms);
        renderAll();
        show('ua-content');
        els.identifier.focus();
      } else {
        if (!perms.view) {
          showDenied("You can't view users",
            'Viewing users needs the Can view SalesAuthUser or Can change SalesAuthUser permission. Ask an administrator to grant it.');
          return;
        }
        if (!(await initEditMode(perms))) return;
        if (!state.readonly) renderAll(); else renderSummary();
        show('ua-content');
      }
    } catch (_) {
      showError();
    }
  }

  init();
})();
