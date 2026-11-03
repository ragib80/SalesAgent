/* Voice of Sales — shared helpers for the user management pages.
   Permissions shown here only drive the UI; every action is enforced by the API. */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function initials(u) {
    const name = (u.full_name || `${u.first_name || ''} ${u.last_name || ''}`).trim();
    const src = name ? name.split(/\s+/).map(s => s[0]).join('') : (u.username || '?').slice(0, 2);
    return src.slice(0, 2).toUpperCase();
  }

  const dateFmt = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
  const dateTimeFmt = new Intl.DateTimeFormat(undefined, {
    day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit',
  });

  function fmtDate(iso, withTime) {
    if (!iso) return '';
    const d = new Date(iso);
    return isNaN(d) ? '' : (withTime ? dateTimeFmt : dateFmt).format(d);
  }

  function fmtRelative(iso) {
    if (!iso) return 'Never';
    const d = new Date(iso);
    if (isNaN(d)) return '';
    const days = Math.floor((Date.now() - d.getTime()) / 86400000);
    if (days < 1) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 30) return `${days} days ago`;
    return fmtDate(iso);
  }

  function errorMessage(data, fallback) {
    if (!data) return fallback;
    if (typeof data.detail === 'string') return data.detail;
    for (const v of Object.values(data)) {
      if (Array.isArray(v) && v.length) return String(v[0]);
      if (typeof v === 'string') return v;
    }
    return fallback;
  }

  /* ── API (JWT with a single shared refresh) ── */
  let refreshing = null;

  function refreshAccess() {
    if (!refreshing) {
      refreshing = fetch(`${apiBase}/user/token/refresh/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh: localStorage.getItem('refresh_token') }),
      })
        .then(r => (r.ok ? r.json() : Promise.reject(r)))
        .then(data => {
          localStorage.setItem('auth_token', data.access);
          if (data.refresh) localStorage.setItem('refresh_token', data.refresh);
          return true;
        })
        .catch(() => false)
        .finally(() => { refreshing = null; });
    }
    return refreshing;
  }

  async function api(path, { method = 'GET', body, params } = {}) {
    let url = `${apiBase}${path}`;
    if (params) {
      const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== '' && v != null));
      if ([...qs].length) url += `?${qs}`;
    }
    const send = () => fetch(url, {
      method,
      headers: {
        Authorization: 'Bearer ' + localStorage.getItem('auth_token'),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    let res = await send();
    if (res.status === 401) {
      if (!(await refreshAccess())) {
        localStorage.clear();
        window.location.replace('/welcome');
        throw new Error('Session expired');
      }
      res = await send();
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* empty body */ }
    return { ok: res.ok, status: res.status, data };
  }

  /* ── Page states ── */
  function show(id) {
    document.querySelectorAll('body > main').forEach(el => { el.hidden = el.id !== id; });
    window.scrollTo(0, 0);
  }

  function showDenied(title, text) {
    $('ua-denied-title').textContent = title;
    $('ua-denied-text').textContent = text;
    show('ua-denied');
  }

  function showError(text, title) {
    if (title) $('ua-error-title').textContent = title;
    if (text) $('ua-error-text').textContent = text;
    show('ua-error');
  }

  /* ── Toast ── */
  let toastTimer = null;
  function toast(message, kind) {
    const el = $('ua-toast');
    el.className = `ua-toast ${kind === 'error' ? 'ua-toast--error' : ''}`;
    el.innerHTML = `<i class="bi ${kind === 'error' ? 'bi-exclamation-circle' : 'bi-check-circle-fill'}" aria-hidden="true"></i>` +
                   `<span>${esc(message)}</span>`;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
  }

  /* ── Theme ── */
  function syncThemeButton() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    const btn = $('theme-toggle');
    btn.innerHTML = `<i class="bi ${dark ? 'bi-sun-fill' : 'bi-moon-fill'}" aria-hidden="true"></i>`;
    btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    btn.title = btn.getAttribute('aria-label');
  }
  $('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (_) { /* ignore */ }
    syncThemeButton();
  });
  syncThemeButton();

  /* ── Permissions + nav ── */
  async function loadPerms() {
    const me = await api('/user/whoami/');
    if (!me.ok || !me.data) throw new Error('whoami failed');
    const perms = me.data.user_management || { view: false, add: false, change: false };
    perms.me = me.data;

    const page = document.body.dataset.page;
    const users = $('ua-nav-users');
    const add = $('ua-nav-add');
    users.hidden = !perms.view;
    add.hidden = !perms.add;
    if (page === 'list') users.setAttribute('aria-current', 'page');
    if (page === 'add') add.setAttribute('aria-current', 'page');
    return perms;
  }

  window.VOS = {
    $, esc, initials, fmtDate, fmtRelative, errorMessage, api, show, showDenied, showError, toast, loadPerms,
  };
})();
