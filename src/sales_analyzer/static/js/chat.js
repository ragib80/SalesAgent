'use strict';

/* ======================================================
   Voice of Sales — Chat UI
   All existing API calls & business logic preserved.
   Added: theme management, welcome screen, message actions.
   ====================================================== */

/* ── Global helpers ── */

function escapeAttr(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function copyMsgText(btn) {
  const text = btn.getAttribute('data-text') || '';
  if (!navigator.clipboard) {
    const el = document.createElement('textarea');
    el.value = text; el.style.position = 'fixed'; el.style.opacity = '0';
    document.body.appendChild(el); el.select();
    try { document.execCommand('copy'); } catch (_) {}
    document.body.removeChild(el);
    _showCopyDone(btn);
    return;
  }
  navigator.clipboard.writeText(text).then(() => _showCopyDone(btn)).catch(() => {});
}

function _showCopyDone(btn) {
  const orig = btn.innerHTML;
  btn.innerHTML = '<i class="bi bi-check2" aria-hidden="true"></i>';
  btn.classList.add('active');
  setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('active'); }, 1500);
}

function voteMsgBtn(btn, dir) {
  const wasActive = btn.classList.contains('active');
  const actions = btn.closest('.message-actions');
  if (actions) {
    actions.querySelectorAll('.msg-action-btn').forEach(b => b.classList.remove('active'));
  }
  if (!wasActive) btn.classList.add('active');
}

function setInputAndFocus(text) {
  const $input = $('#message-input');
  $input.val(text).trigger('input').focus();
  const el = $input[0];
  if (el) { el.style.height = ''; el.style.height = el.scrollHeight + 'px'; }
  $('#send-btn').prop('disabled', !text.trim());
}

/* ── Theme management ── */

function applyTheme(theme) {
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const resolved = theme === 'auto' ? (prefersDark ? 'dark' : 'light') : theme;
  document.documentElement.setAttribute('data-theme', resolved);
  localStorage.setItem('theme', theme);
  _syncThemeUI();
}

function _syncThemeUI() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const $btn = $('#theme-toggle');
  if ($btn.length) {
    $btn.html(isDark
      ? '<i class="bi bi-sun-fill" aria-hidden="true"></i>'
      : '<i class="bi bi-moon-fill" aria-hidden="true"></i>');
    $btn.attr('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    $btn.attr('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  }
  const saved = localStorage.getItem('theme') || 'auto';
  $(`input[name="theme"][value="${saved}"]`).prop('checked', true);
}

function initTheme() {
  const saved = localStorage.getItem('theme') || 'auto';
  applyTheme(saved);
}

/* ── Mobile sidebar ── */
$(document).ready(function () {
  $('#mobile-menu-btn').on('click', function () {
    const $sidebar = $('#sidebar');
    const $overlay = $('#sidebar-overlay');
    if ($sidebar.hasClass('open')) {
      $sidebar.removeClass('open');
      $overlay.removeClass('show');
    } else {
      $sidebar.addClass('open');
      $overlay.addClass('show');
    }
  });

  $('#sidebar-overlay').on('click', function () {
    $('#sidebar').removeClass('open');
    $('#sidebar-overlay').removeClass('show');
  });

  $('.chats-section').on('click', '.chat-item', function () {
    if (window.innerWidth <= 768) {
      $('#sidebar').removeClass('open');
      $('#sidebar-overlay').removeClass('show');
    }
  });

  $('.sidebar-menu .menu-item:contains("New chat")').on('click', function () {
    if (window.innerWidth <= 768) {
      $('#sidebar').removeClass('open');
      $('#sidebar-overlay').removeClass('show');
    }
  });
});

/* ── Model dropdown ── */
window.onModelSelected = function (selector) {
  console.log('Selected:', selector);
};

document.addEventListener('DOMContentLoaded', () => {
  const menu   = document.getElementById('modelMenu');
  const label  = document.getElementById('current-model');
  const toggle = document.getElementById('modelDropdown');
  if (!menu) return;

  menu.addEventListener('click', (e) => {
    const item = e.target.closest('.dropdown-item[data-selector]');
    if (!item) return;
    label.textContent = item.textContent.trim();
    window.onModelSelected(item.dataset.selector);
    bootstrap.Dropdown.getOrCreateInstance(toggle).hide();
  });
});

/* ── Main chat logic ── */
$(function () {
  let isSubmitting = false;
  let typingIndicator = null;

  let chatsPage    = 1;
  let chatsHasNext = true;
  let chatsIsLoading = false;

  let msgsIsLoading = false;
  let msgsHasNext   = false;
  let msgsNextPath  = null;

  // User info
  let fullName    = localStorage.getItem('full_name')    || 'User';
  let userName    = localStorage.getItem('username')     || 'user';
  let designation = localStorage.getItem('designation')  || '';

  $('.user-info .user-name').text(fullName);
  $('.user-info .user-plan').text(designation);

  const initials = (fullName.trim()
    ? fullName.trim().split(/\s+/).map(s => s[0]).join('')
    : userName.slice(0, 2)
  ).slice(0, 2).toUpperCase();

  $('.user-info .user-avatar').text(initials || 'U');

  /* ── Theme wiring ── */
  initTheme();
  _syncThemeUI();

  $('#theme-toggle').on('click', function () {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  $('input[name="theme"]').on('change', function () {
    applyTheme($(this).val());
  });

  /* ── Character counter ── */
  const MAX_CHARS = 2000;
  $('#message-input').on('input', function () {
    const len = $(this).val().length;
    const $counter = $('#char-counter');
    if (len > 200) {
      $counter.text(len > MAX_CHARS ? `${len}/${MAX_CHARS}` : len)
              .removeClass('warning overflow')
              .addClass(len > MAX_CHARS ? 'overflow' : (len > MAX_CHARS * 0.8 ? 'warning' : ''))
              .addClass('visible');
    } else {
      $counter.removeClass('visible warning overflow').text('');
    }
  });

  /* ── Auth helpers ── */
  function getAuthToken() { return localStorage.getItem('auth_token'); }

  function refreshToken() {
    $.ajax({
      url: apiBase + '/user/token/refresh/',
      method: 'POST',
      data: JSON.stringify({ refresh: localStorage.getItem('refresh_token') }),
      contentType: 'application/json',
      success(resp) {
        localStorage.setItem('auth_token', resp.access);
        localStorage.setItem('refresh_token', resp.refresh);
        fetchChats(false);
      },
      error() {
        localStorage.clear();
        window.location.href = '/welcome';
      },
    });
  }

  function sendAuthenticatedRequest(url, method, data, onSuccess, onError) {
    const token = getAuthToken();
    if (!token) { window.location.href = '/welcome'; return; }
    $.ajax({
      url: apiBase + url,
      method,
      headers: { Authorization: 'Bearer ' + token },
      data,
      contentType: 'application/json',
      success: onSuccess,
      error(xhr, status, err) {
        if (xhr.status === 401) refreshToken();
        else if (onError) onError(xhr, status, err);
      },
    });
  }

  function scrollToBottom() {
    const c = document.getElementById('chat-content');
    if (c) c.scrollTop = c.scrollHeight;
  }

  function updateSendButton() {
    const text = $('#message-input').val().trim();
    $('#send-btn').prop('disabled', text.length === 0 || isSubmitting);
  }

  /* ── Welcome screen ── */
  function renderWelcomeScreen() {
    const suggestions = [
      { bi: 'bi-bar-chart-fill',   bg: 'rgba(16,163,127,.12)',  color: '#10a37f', text: 'Top 10 dealers by revenue this month',       sub: 'Performance ranking' },
      { bi: 'bi-graph-down-arrow', bg: 'rgba(239,68,68,.1)',    color: '#ef4444', text: 'Which dealers have declining sales vs last year?', sub: 'Growth & decline analysis' },
      { bi: 'bi-trophy-fill',      bg: 'rgba(245,158,11,.12)',  color: '#f59e0b', text: 'Best performing brands this fiscal year',     sub: 'Brand-level summary' },
      { bi: 'bi-calendar3',        bg: 'rgba(99,102,241,.12)',  color: '#6366f1', text: 'Monthly sales trend for the last 6 months',   sub: 'Trend analysis' },
    ];

    const cardsHtml = suggestions.map(s => `
      <div class="welcome-card" role="button" tabindex="0"
           onclick="setInputAndFocus('${escapeAttr(s.text)}')"
           onkeydown="if(event.key==='Enter'||event.key===' '){setInputAndFocus('${escapeAttr(s.text)}')}"
           aria-label="${escapeAttr(s.text)}">
        <div class="welcome-card-icon-wrap" style="background:${s.bg};" aria-hidden="true">
          <i class="bi ${s.bi}" style="color:${s.color}; font-size:17px;"></i>
        </div>
        <div class="welcome-card-text">${s.text}</div>
        <div class="welcome-card-sub">${s.sub}</div>
      </div>
    `).join('');

    return `
      <div class="welcome-screen" role="main">
        <div class="welcome-logo" aria-hidden="true">
          <i class="bi bi-bar-chart-line-fill"></i>
        </div>
        <div class="welcome-eyebrow">Sales Intelligence</div>
        <h1 class="welcome-title">Voice of Sales</h1>
        <p class="welcome-subtitle">
          Your AI assistant for SAP sales data.<br>Ask in plain language - get instant insights.
        </p>
        <div class="welcome-cards" role="list" aria-label="Suggested prompts">
          ${cardsHtml}
        </div>
      </div>
    `;
  }

  /* ── Chats (sidebar) ── */
  function renderLoadMoreButton() {
    const $sec = $('.chats-section');
    const id = 'load-more-convos';
    $('#' + id).remove();
    if (chatsHasNext) {
      $sec.append(`
        <div id="${id}" class="menu-item load-more-btn" role="button" tabindex="0">
          <i class="bi bi-plus-circle" style="fill:none;"></i> Load more
        </div>
      `);
    }
  }

  function setLoadMoreLoading(loading) {
    const $btn = $('#load-more-convos');
    if (!$btn.length) return;
    if (loading) {
      $btn.addClass('disabled').css('pointer-events', 'none')
          .html('<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Loading…');
    } else {
      $btn.removeClass('disabled').css('pointer-events', '')
          .html('<i class="bi bi-plus-circle" style="fill:none;"></i> Load more');
    }
  }

  function fetchChats(append = false, cb) {
    if (chatsIsLoading) return;

    const $sec = $('.chats-section');

    if (!append) {
      chatsPage    = 1;
      chatsHasNext = true;
      $sec.empty().append('<div class="chats-header">Chats</div>');
    } else {
      setLoadMoreLoading(true);
    }

    if (!chatsHasNext) {
      renderLoadMoreButton();
      if (cb) cb();
      return;
    }

    chatsIsLoading = true;

    sendAuthenticatedRequest(
      `/conversations/?page=${chatsPage}`, 'GET', null,
      (resp) => {
        chatsIsLoading = false;
        setLoadMoreLoading(false);

        const meta = resp && (resp.meta
          ? resp.meta
          : (typeof resp === 'object' && ('next' in resp || 'previous' in resp || 'count' in resp))
            ? { next: resp.next, previous: resp.previous, count: resp.count }
            : null);

        const list = resp && resp.results ? resp.results : Array.isArray(resp) ? resp : [];

        if (!append && list.length === 0) {
          $sec.append('<div class="no-conversations">No previous conversations</div>');
        } else {
          list.forEach((c) => {
            const active = c.uuid === currentChatId ? 'active' : '';
            if ($sec.find(`.chat-item[data-chat-id="${c.uuid}"]`).length) return;
            $sec.append(`
              <div class="chat-item ${active}" data-chat-id="${c.uuid}" role="button" tabindex="0">
                <div class="chat-title">${c.title || 'Untitled Chat'}</div>
                <button class="chat-menu-btn" onclick="toggleChatDropdown(event,'${c.uuid}')"
                        aria-label="Chat options" aria-haspopup="true">
                  <i class="bi bi-three-dots" aria-hidden="true" style="fill:none;"></i>
                </button>
                <div class="chat-dropdown" id="dropdown-${c.uuid}" role="menu">
                  <button class="dropdown-item" onclick="shareChat('${c.uuid}')" role="menuitem">
                    <i class="bi bi-share dropdown-icon" style="fill:none;"></i>Share
                  </button>
                  <button class="dropdown-item" onclick="renameChat('${c.uuid}')" role="menuitem">
                    <i class="bi bi-pencil dropdown-icon" style="fill:none;"></i>Rename
                  </button>
                  <button class="dropdown-item" onclick="archiveChat('${c.uuid}')" role="menuitem">
                    <i class="bi bi-archive dropdown-icon" style="fill:none;"></i>Archive
                  </button>
                  <button class="dropdown-item delete" onclick="deleteChat('${c.uuid}')" role="menuitem">
                    <i class="bi bi-trash3 dropdown-icon" style="fill:none;"></i>Delete
                  </button>
                </div>
              </div>
            `);
          });
        }

        if (meta) {
          chatsHasNext = !!meta.next;
          if (meta.next) {
            const m = /[?&]page=(\d+)/.exec(meta.next);
            chatsPage = m ? parseInt(m[1], 10) : chatsPage + 1;
          }
        } else {
          chatsHasNext = false;
        }

        renderLoadMoreButton();
        if (cb) cb();
      },
      () => {
        chatsIsLoading = false;
        setLoadMoreLoading(false);
        if (!append) {
          $sec.append('<div class="no-conversations text-danger">Failed to load conversations.</div>');
        }
        renderLoadMoreButton();
        if (cb) cb();
      }
    );
  }

  /* ── Messages utilities ── */
  function ensureMessageShell() {
    const $chat = $('#chat-content');
    if (!$('#messages-list').length) {
      $chat.html(`
        <div id="load-prev-wrap" class="text-center"></div>
        <div id="messages-list"></div>
      `);
    }
  }

  function showInitialMessagesLoader() {
    const $list = $('#messages-list');
    if (!$list.length || $('#initial-msgs-loader').length) return;
    $list.append(`
      <div id="initial-msgs-loader" class="py-3 d-flex justify-content-center">
        <div class="spinner-border" role="status" aria-label="Loading messages"></div>
      </div>
    `);
  }
  function hideInitialMessagesLoader() { $('#initial-msgs-loader').remove(); }

  function renderLoadPrevButton() {
    const $wrap = $('#load-prev-wrap');
    if (!$wrap.length) return;
    if (!msgsHasNext) { $wrap.empty(); return; }
    if (!$('#load-prev-msgs').length) {
      $wrap.html(`
        <button id="load-prev-msgs"
                class="btn btn-outline-secondary btn-sm d-none"
                style="margin:6px auto 12px; display:inline-flex; align-items:center; gap:.4rem;">
          <span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span>
          Load previous
        </button>
      `);
    }
    updateLoadPrevVisibility();
  }

  function setLoadPrevLoading(loading) {
    const $btn = $('#load-prev-msgs');
    if (!$btn.length) return;
    $btn.prop('disabled', loading);
    $btn.find('.spinner-border').toggleClass('d-none', !loading);
  }

  function computeTopThresholdPx() {
    const c = document.getElementById('chat-content');
    if (!c) return 16;
    return Math.max(8, Math.min(64, Math.round(c.clientHeight * 0.025)));
  }
  function isNearTop() {
    const c = document.getElementById('chat-content');
    return c ? c.scrollTop <= computeTopThresholdPx() : false;
  }
  function updateLoadPrevVisibility() {
    const $btn = $('#load-prev-msgs');
    if (!$btn.length) return;
    if (!msgsHasNext) { $btn.addClass('d-none'); return; }
    if (isNearTop()) $btn.removeClass('d-none'); else $btn.addClass('d-none');
  }

  function resolveRole(m) {
    const raw = (m.sender ?? m.role ?? m.author ?? '').toString().toLowerCase();
    if (raw === 'user' || raw === 'assistant') return raw;
    if (typeof m.is_user === 'boolean') return m.is_user ? 'user' : 'assistant';
    if (m.sender === userName) return 'user';
    return 'assistant';
  }

  function getText(m) { return m.text ?? m.content ?? m.message ?? ''; }

  /* Message action bar HTML */
  function _actionsHtml(rawText) {
    return `
      <div class="message-actions" role="toolbar" aria-label="Message actions">
        <button class="msg-action-btn" title="Copy response"
                onclick="copyMsgText(this)"
                data-text="${escapeAttr(rawText)}"
                aria-label="Copy response">
          <i class="bi bi-clipboard" aria-hidden="true"></i>
        </button>
        <button class="msg-action-btn" title="Good response"
                onclick="voteMsgBtn(this,'up')"
                aria-label="Good response">
          <i class="bi bi-hand-thumbs-up" aria-hidden="true"></i>
        </button>
        <button class="msg-action-btn" title="Bad response"
                onclick="voteMsgBtn(this,'down')"
                aria-label="Bad response">
          <i class="bi bi-hand-thumbs-down" aria-hidden="true"></i>
        </button>
      </div>
    `;
  }

  /* Single message renderer — wraps in .msg-row for hover-actions */
  function renderMessageEl(m) {
    const role   = resolveRole(m);
    const cls    = role === 'user' ? 'user' : 'assistant';
    const avatar = role === 'assistant'
      ? '<i class="bi bi-stars" aria-hidden="true"></i>'
      : '';
    const rawText = getText(m);
    const html    = marked.parse(rawText);

    if (role === 'assistant') {
      const historyToggle = (m.has_data && m.id)
        ? `<div class="vis-accordions vis-history-mode" data-message-id="${m.id}">
             <div class="vis-accordion vis-accordion-offcanvas" data-type="chart">
               <button class="vis-accordion-header">
                 <i class="bi bi-bar-chart" aria-hidden="true"></i>
                 <span>Chart</span>
                 <i class="bi bi-box-arrow-right vis-panel-hint" aria-hidden="true"></i>
               </button>
             </div>
             <div class="vis-accordion" data-type="table">
               <div class="vis-accordion-header-row">
                 <button class="vis-accordion-header">
                   <i class="bi bi-table" aria-hidden="true"></i>
                   <span>Table</span>
                   <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
                 </button>
                 <button class="vis-export-btn" data-message-id="${m.id}" title="Export all records to Excel">
                   <i class="bi bi-file-earmark-excel-fill" aria-hidden="true"></i>
                   <span>Export</span>
                 </button>
               </div>
               <div class="vis-accordion-body" style="display:none;"></div>
             </div>
           </div>`
        : '';

      return `
        <div class="msg-row">
          <div class="message ${cls}">
            <div class="message-avatar ${cls}" aria-hidden="true">${avatar}</div>
            <div class="message-content">${html}</div>
          </div>
          ${_actionsHtml(rawText)}
          ${historyToggle}
        </div>
      `;
    }

    return `
      <div class="msg-row">
        <div class="message ${cls}">
          <div class="message-avatar ${cls}" aria-hidden="true"></div>
          <div class="message-content">${html}</div>
        </div>
      </div>
    `;
  }

  function deriveNextPathFromMeta(meta, chatId) {
    if (meta && meta.has_next) {
      if (meta.next_page) return `/conversations/${chatId}/messages/?page=${meta.next_page}`;
      if (meta.next) {
        try {
          const u = new URL(meta.next);
          let path = u.pathname + u.search;
          if (path.startsWith('/api/')) path = path.slice(4);
          return path;
        } catch { return meta.next.replace(/^\/api\//, '/'); }
      }
      return `/conversations/${chatId}/messages/?page=2`;
    }
    if (meta && meta.next) {
      try {
        const u = new URL(meta.next);
        let path = u.pathname + u.search;
        if (path.startsWith('/api/')) path = path.slice(4);
        return path;
      } catch { return meta.next.replace(/^\/api\//, '/'); }
    }
    return null;
  }

  function fetchMessages(chatId, mode = 'reset') {
    const $chat = $('#chat-content');

    if (mode === 'reset') {
      msgsIsLoading = false;
      msgsHasNext   = false;
      msgsNextPath  = null;
      ensureMessageShell();
      $('#messages-list').empty();
      $('#load-prev-wrap').empty();
      showInitialMessagesLoader();
    }

    if (msgsIsLoading) return;
    msgsIsLoading = true;
    if (mode === 'prepend') setLoadPrevLoading(true);

    const url = (mode === 'prepend' && msgsNextPath)
      ? msgsNextPath
      : `/conversations/${chatId}/messages/?page=1`;

    sendAuthenticatedRequest(url, 'GET', null, (resp) => {
      msgsIsLoading = false;
      if (mode === 'prepend') setLoadPrevLoading(false);

      const meta = resp && (resp.meta ? resp.meta : (
        (typeof resp === 'object' && ('next' in resp || 'previous' in resp || 'count' in resp))
          ? { next: resp.next, previous: resp.previous, count: resp.count }
          : null
      ));

      const list = resp && resp.results ? resp.results : Array.isArray(resp) ? resp : [];
      const $list = $('#messages-list');

      if (mode === 'reset') hideInitialMessagesLoader();

      if (mode === 'reset' && list.length === 0) {
        $chat.html('<div class="text-center text-muted py-4">No messages yet. Start the conversation!</div>');
        return;
      }

      const ascending = list.slice().sort((a, b) => {
        const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
        const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
        if (ta !== tb) return ta - tb;
        return (a.id || 0) - (b.id || 0);
      });

      if (mode === 'reset') {
        $list.empty();
        const frag = document.createDocumentFragment();
        const tmp  = document.createElement('div');
        ascending.forEach(m => {
          tmp.innerHTML = renderMessageEl(m);
          while (tmp.firstChild) frag.appendChild(tmp.firstChild);
        });
        $list[0].appendChild(frag);
        const c = document.getElementById('chat-content');
        c.scrollTop = c.scrollHeight;
      } else {
        const c = document.getElementById('chat-content');
        const before = c.scrollHeight;
        const prevScrollTop = c.scrollTop;
        const frag = document.createDocumentFragment();
        const tmp  = document.createElement('div');
        ascending.forEach(m => {
          tmp.innerHTML = renderMessageEl(m);
          while (tmp.firstChild) frag.appendChild(tmp.firstChild);
        });
        $list[0].insertBefore(frag, $list[0].firstChild);
        const after = c.scrollHeight;
        c.scrollTop = prevScrollTop + (after - before);
      }

      const nextPath = deriveNextPathFromMeta(meta, chatId);
      msgsHasNext  = !!nextPath;
      msgsNextPath = nextPath;

      renderLoadPrevButton();
      updateLoadPrevVisibility();

    }, () => {
      msgsIsLoading = false;
      if (mode === 'prepend') setLoadPrevLoading(false);
      if (mode === 'reset') {
        hideInitialMessagesLoader();
        $chat.html('<div class="text-center text-danger py-4">Failed to load messages.</div>');
      }
    });
  }

  function startNewChat() {
    currentChatId = null;
    $('#chat-id-holder').attr('data-current-conversation-id', '').data('current-conversation-id', '');
    $('.chat-item').removeClass('active');
    $('#chat-content').empty().append(renderWelcomeScreen());
  }

  function selectChat(chatId) {
    currentChatId = chatId;
    $('.chat-item').removeClass('active');
    $(`.chat-item[data-chat-id="${chatId}"]`).addClass('active');
    fetchMessages(chatId, 'reset');
    $('#chat-id-holder')
      .attr('data-current-conversation-id', chatId)
      .data('current-conversation-id', chatId);
  }

  /* ── Streaming ── */
  let streamingText    = '';
  let $streamBubble    = null;
  let _rafPending      = false;
  let $msgList         = null;  // the list element that holds THIS exchange's bubbles
  let $lastUserBubble  = null;  // direct ref to the user bubble for THIS exchange

  /* ── Visualization state (per SSE exchange) ── */
  let _pendingChartData = null;   // {cols, rows, total_rows, message_id} from data event
  const _visStore = new WeakMap(); // DOM element → {cols, rows, total_rows, messageId}

  /* Return the live (in-document) message list, falling back gracefully. */
  function _liveList() {
    if ($msgList && $.contains(document.body, $msgList[0])) return $msgList;
    return $('#messages-list').length ? $('#messages-list') : $('#chat-content');
  }

  /* After finalize, guarantee user bubble is directly before the AI bubble. */
  function _fixOrder() {
    const $c = $('#messages-list');
    if (!$c.length) return;
    const $rows  = $c.children('.msg-row');
    if ($rows.length < 2) return;
    const $aiRow   = $rows.filter((_, el) => !!$(el).find('.message.assistant').length).last();
    const $userRow = $rows.filter((_, el) => !!$(el).find('.message.user').length).last();
    if (!$aiRow.length || !$userRow.length) return;
    if ($rows.index($aiRow[0]) < $rows.index($userRow[0])) {
      $aiRow.insertAfter($userRow);
    }
  }

  function _ensureStreamBubble() {
    if ($streamBubble && $streamBubble.length) return;
    $streamBubble = $(`
      <div class="msg-row">
        <div class="message assistant">
          <div class="message-avatar assistant" aria-hidden="true">
            <i class="bi bi-stars"></i>
          </div>
          <div class="message-content stream-content"></div>
        </div>
      </div>`);
    if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
    // Insert stream bubble directly AFTER the user bubble so order is always correct.
    if ($lastUserBubble && $.contains(document.body, $lastUserBubble[0])) {
      $lastUserBubble.after($streamBubble);
    } else {
      _liveList().append($streamBubble);
    }
  }

  function _scheduleStreamRender() {
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(function () {
      _rafPending = false;
      if (!$streamBubble || !$streamBubble.length) return;
      const $c = $streamBubble.find('.stream-content');
      if (!$c.length) return;
      $c.html(marked.parse(streamingText) + '<span class="stream-cursor"></span>');
      scrollToBottom();
    });
  }

  function _sseFinalize(answer, uuid) {
    _rafPending = false;
    const $live = _liveList();

    // Recover user bubble if it ended up detached (e.g. DOM was rebuilt mid-stream)
    if ($lastUserBubble && !$.contains(document.body, $lastUserBubble[0])) {
      $live.append($lastUserBubble);
    }

    if ($streamBubble && $streamBubble.length) {
      // Recover detached stream bubble
      if (!$.contains(document.body, $streamBubble[0])) {
        $live.append($streamBubble);
      }
      const finalText = answer || streamingText;
      $streamBubble.find('.stream-content').html(marked.parse(finalText));
      $streamBubble.append(_actionsHtml(finalText));
      $streamBubble = null;
    } else {
      if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
      $live.append(renderMessageEl({ sender: 'assistant', text: answer }));
    }

    // Final safety net: ensure user bubble is always before AI bubble
    _fixOrder();

    // Attach visualization toggles if we received a data event for this exchange
    if (_pendingChartData) {
      const chartData = _pendingChartData;
      _pendingChartData = null;
      const $aiRow = _liveList().children('.msg-row').filter((_, el) =>
        !!$(el).find('.message.assistant').length
      ).last();
      if ($aiRow.length) _attachVisToggles($aiRow, chartData);
    }

    streamingText   = '';
    isSubmitting    = false;
    updateSendButton();
    $('#message-input').prop('disabled', false).focus();
    requestAnimationFrame(() => scrollToBottom());

    if (!currentChatId && uuid) {
      currentChatId = uuid;
      fetchChats(false, () => {
        $('.chat-item').removeClass('active');
        $(`.chat-item[data-chat-id="${currentChatId}"]`).addClass('active');
        $('#chat-id-holder')
          .attr('data-current-conversation-id', currentChatId)
          .data('current-conversation-id', currentChatId);
      });
    }
  }

  /* ── Send message (SSE streaming) ── */
  async function sendMessage() {
    const $input = $('#message-input');
    const text   = $input.val().trim();
    if (!text || isSubmitting) return;

    const token = getAuthToken();
    if (!token) { window.location.href = '/welcome'; return; }

    isSubmitting    = true;
    streamingText   = '';
    $streamBubble   = null;
    $lastUserBubble = null;
    $msgList        = null;
    $input.val('').css('height', 'auto').prop('disabled', true);
    updateSendButton();
    $('#char-counter').removeClass('visible warning overflow').text('');

    // User bubble — capture $msgList and $lastUserBubble so every stream helper uses them
    ensureMessageShell();
    $msgList = $('#messages-list').length ? $('#messages-list') : $('#chat-content');
    $msgList.append(renderMessageEl({ sender: userName, text }));
    $lastUserBubble = $msgList.children('.msg-row').last();
    scrollToBottom();

    // Typing indicator
    typingIndicator = $(`
      <div class="msg-row" id="sse-typing-row">
        <div class="message assistant">
          <div class="message-avatar assistant" aria-hidden="true">
            <i class="bi bi-stars"></i>
          </div>
          <div class="message-content">
            <div class="typing-indicator" aria-label="AI is thinking">
              <span class="dot"></span>
              <span class="dot"></span>
              <span class="dot"></span>
            </div>
            <div id="sse-status"
                 style="font-size:.8rem; color:var(--vs-text-muted); margin-top:4px; min-height:1em;"
                 aria-live="polite"></div>
          </div>
        </div>
      </div>`);
    $msgList.append(typingIndicator);
    scrollToBottom();

    const body = { prompt: text };
    if (currentChatId) body.conversation_id = currentChatId;

    try {
      const resp = await fetch(apiBase + '/sales/query/stream/', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer ' + token,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        if (resp.status === 401) { refreshToken(); return; }
        throw new Error('HTTP ' + resp.status);
      }

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf       = '';
      let eventName = 'message';
      let dataStr   = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop();

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataStr = line.slice(5).trim();
          } else if (line === '') {
            if (!dataStr) { eventName = 'message'; dataStr = ''; continue; }
            let payload;
            try { payload = JSON.parse(dataStr); } catch { eventName = 'message'; dataStr = ''; continue; }

            if (eventName === 'status') {
              if (typingIndicator) typingIndicator.find('#sse-status').text(payload.message || '');
              scrollToBottom();
            } else if (eventName === 'token') {
              const chunk = payload.chunk || '';
              if (chunk) {
                streamingText += chunk;
                _ensureStreamBubble();
                _scheduleStreamRender();
              }
            } else if (eventName === 'data') {
              if (payload.cols && Array.isArray(payload.rows)) {
                _pendingChartData = {
                  cols: payload.cols,
                  col_labels: payload.col_labels || payload.cols,
                  rows: payload.rows,
                  total_rows: payload.total_rows || payload.rows.length,
                  message_id: payload.message_id || null,
                  chart_meta: payload.chart_meta || null,
                };
              }
            } else if (eventName === 'final') {
              const answer = (payload.answer || streamingText || '').trim()
                || 'I prepared the results for you — see details below.';
              if (_pendingChartData && payload.message_id && !_pendingChartData.message_id) {
                _pendingChartData.message_id = payload.message_id;
              }
              _sseFinalize(answer, payload.uuid);
            } else if (eventName === 'error') {
              _pendingChartData = null;
              const msg = (payload.message || 'Something went wrong.').trim();
              _sseFinalize(msg, payload.uuid);
            }

            eventName = 'message';
            dataStr   = '';
          }
        }
      }

      if (isSubmitting) {
        _sseFinalize(
          streamingText || 'I prepared the results — please check the conversation.',
          null
        );
      }

    } catch (err) {
      if ($streamBubble) { $streamBubble.remove(); $streamBubble = null; }
      if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
      streamingText = '';
      isSubmitting  = false;
      updateSendButton();
      $input.prop('disabled', false).focus();
      Swal.fire({ icon: 'error', title: 'Connection error', text: 'Failed to reach the server. Please try again.' });
    }
  }

  /* ── Event bindings ── */
  $('#send-btn').on('click', () => sendMessage());

  $('#message-input').on('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  $('#message-input').on('input', function () {
    this.style.height = '';
    this.style.height = this.scrollHeight + 'px';
    updateSendButton();
  });

  $('.chats-section').on('click', '.chat-item', function () {
    selectChat($(this).data('chat-id'));
  });

  $('.sidebar-menu .menu-item:contains("New chat")').on('click', startNewChat);

  $('.chats-section').on('click', '#load-more-convos', function () { fetchChats(true); });

  $('#chat-content').on('click', '#load-prev-msgs', function () {
    if (currentChatId && msgsHasNext && !msgsIsLoading) fetchMessages(currentChatId, 'prepend');
  });

  let topScrollTimer = null;
  const _chatContentEl = document.getElementById('chat-content');
  if (_chatContentEl) {
    _chatContentEl.addEventListener('scroll', function () {
      if (topScrollTimer) clearTimeout(topScrollTimer);
      topScrollTimer = setTimeout(updateLoadPrevVisibility, 50);
    }, { passive: true });
  }

  let resizeTimer = null;
  $(window).on('resize', function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(updateLoadPrevVisibility, 100);
  });

  /* ── Search chats ── */
  $('#search-chats-btn').on('click', function () {
    const $wrap = $('#chat-search-wrap');
    $wrap.toggle();
    if ($wrap.is(':visible')) {
      $('#chat-search-input').val('').trigger('input').focus();
    } else {
      $('.chats-section .chat-item').show();
    }
  });

  $('#chat-search-input').on('input', function () {
    const q = $(this).val().trim().toLowerCase();
    $('.chats-section .chat-item').each(function () {
      const title = $(this).find('.chat-title').text().toLowerCase();
      $(this).toggle(!q || title.includes(q));
    });
  });

  /* ── Initialise ── */
  const $sec = $('.chats-section');
  if ($sec.children('.chats-header').length === 0) {
    $sec.prepend('<div class="chats-header">Chats</div>');
  }

  fetchChats(false);
  startNewChat();
  updateSendButton();

  /* ── Visualization: chart → offcanvas, table → inline accordion ── */
  $(document).on('click', '.vis-accordion-header', function () {
    const $header = $(this);
    const $accordion = $header.closest('.vis-accordion');
    const $wrap = $accordion.closest('.vis-accordions');
    const type = $accordion.data('type');

    if (type === 'chart') {
      _openChartOffcanvas($wrap);
      return;
    }

    // Table: expand / collapse in-place
    const $body = $accordion.find('.vis-accordion-body');
    const isOpen = $body.is(':visible');
    $body.toggle(!isOpen);
    $header.toggleClass('open', !isOpen);
    $header.closest('.vis-accordion-header-row').toggleClass('open', !isOpen);

    if (!isOpen && !$body.data('loaded')) {
      $body.data('loaded', true);

      const messageId = $wrap.data('vis-uid') || $wrap.data('message-id');
      window._visStore = window._visStore || new WeakMap();
      const preview = window._visStore.get($wrap[0]);

      // Prefer full paginated table when a numeric message_id is available (consistent with reload)
      const rawMid = messageId || (preview && preview.message_id);
      const tableMessageId = rawMid && /^\d+$/.test(String(rawMid)) ? rawMid : null;

      if (tableMessageId) {
        _loadFullTable($body[0], tableMessageId);
      } else if (preview) {
        _renderPreviewTable($body[0], preview.col_labels || preview.cols, preview.rows, preview.total_rows, null);
      } else {
        $body.html('<p class="vis-error">No table data available.</p>');
      }
    }
  });

  /* ── Export to Excel ── */
  $(document).on('click', '.vis-export-btn', function (e) {
    e.stopPropagation();
    const $btn = $(this);
    const messageId = $btn.data('message-id');
    if (!messageId) return;

    const token = getAuthToken();
    if (!token) { window.location.href = '/welcome'; return; }

    if ($btn.data('exporting')) return; // prevent duplicate click while in-flight

    $btn.data('exporting', true);
    const $icon = $btn.find('i');
    const $label = $btn.find('span');
    $btn.prop('disabled', true);
    $icon.removeClass('bi-file-earmark-excel-fill').addClass('bi-hourglass-split');
    $label.text('Preparing…');

    fetch(`${apiBase}/sales/export-excel/?message_id=${messageId}`, {
      headers: { 'Authorization': 'Bearer ' + token },
    })
    .then(r => {
      if (!r.ok) return r.json().then(d => { throw new Error(d.error || 'Export failed (HTTP ' + r.status + ')'); });
      const disposition = r.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="([^"]+)"/);
      const filename = match ? match[1] : `sales_export_${messageId}.xlsx`;
      return r.blob().then(blob => ({ blob, filename }));
    })
    .then(({ blob, filename }) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    })
    .catch(err => {
      Swal.fire({ icon: 'error', title: 'Export failed', text: err.message || 'Could not generate Excel file. Please try again.' });
    })
    .finally(() => {
      $btn.data('exporting', false).prop('disabled', false);
      $icon.removeClass('bi-hourglass-split').addClass('bi-file-earmark-excel-fill');
      $label.text('Export');
    });
  });
});

/* ── Chart offcanvas: open right-side panel for chart ── */
function _openChartOffcanvas($wrap) {
  const messageId = $wrap.data('vis-uid') || $wrap.data('message-id');
  window._visStore = window._visStore || new WeakMap();
  const preview = window._visStore.get($wrap[0]);

  const offcanvasEl = document.getElementById('chartOffcanvas');
  if (!offcanvasEl) return;

  // If the offcanvas is already visible, shown.bs.offcanvas won't fire again — render immediately
  if (offcanvasEl.classList.contains('show')) {
    const container = document.getElementById('chartOffcanvasContainer');
    if (container) {
      if (window._chartOffcanvasInstance) {
        try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
        window._chartOffcanvasInstance = null;
      }
      container.innerHTML = '';
      if (preview) {
        _renderChart(container, preview.cols, preview.rows, undefined, preview.chart_meta);
      } else if (messageId) {
        _loadChartFromApi(container, messageId);
      } else {
        container.innerHTML = '<p class="vis-error">No chart data available.</p>';
      }
    }
    return;
  }

  window._chartOffcanvasPending = { preview, messageId };
  bootstrap.Offcanvas.getOrCreateInstance(offcanvasEl).show();
}

/* Render chart after offcanvas animation finishes (container has real dimensions) */
(function () {
  document.addEventListener('DOMContentLoaded', function () {
    const offcanvasEl = document.getElementById('chartOffcanvas');
    if (!offcanvasEl) return;

    offcanvasEl.addEventListener('shown.bs.offcanvas', function () {
      const pending = window._chartOffcanvasPending;
      window._chartOffcanvasPending = null;
      if (!pending) return;

      const container = document.getElementById('chartOffcanvasContainer');
      if (!container) return;

      // Dispose previous chart instance before creating a new one
      if (window._chartOffcanvasInstance) {
        try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
        window._chartOffcanvasInstance = null;
      }
      container.innerHTML = '';

      if (pending.preview) {
        _renderChart(container, pending.preview.cols, pending.preview.rows, undefined, pending.preview.chart_meta);
      } else if (pending.messageId) {
        _loadChartFromApi(container, pending.messageId);
      } else {
        container.innerHTML = '<p class="vis-error">No chart data available.</p>';
      }
    });
  });

  // Resize ECharts when window resizes while offcanvas is open
  window.addEventListener('resize', function () {
    if (window._chartOffcanvasInstance) {
      try { window._chartOffcanvasInstance.resize(); } catch (_) {}
    }
  });
})();

/* ═══════════════════════════════════════════════════════
   Visualization helpers (outside $(function) — global scope)
   ═══════════════════════════════════════════════════════ */

/* Attach two accordions (Chart + Table) after a live assistant message row */
function _attachVisToggles($msgRow, chartData) {
  const { cols, col_labels, rows, total_rows, message_id, chart_meta } = chartData;
  const uid = message_id || ('vis-' + Date.now());

  const $accordions = $(`
    <div class="vis-accordions" data-vis-uid="${uid}">
      <div class="vis-accordion vis-accordion-offcanvas" data-type="chart">
        <button class="vis-accordion-header">
          <i class="bi bi-bar-chart" aria-hidden="true"></i>
          <span>Chart</span>
          <i class="bi bi-box-arrow-right vis-panel-hint" aria-hidden="true"></i>
        </button>
      </div>
      <div class="vis-accordion" data-type="table">
        <div class="vis-accordion-header-row">
          <button class="vis-accordion-header">
            <i class="bi bi-table" aria-hidden="true"></i>
            <span>Table</span>
            <span class="vis-accordion-count">${total_rows} records</span>
            <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
          </button>
          ${message_id ? `<button class="vis-export-btn" data-message-id="${message_id}" title="Export all records to Excel">
            <i class="bi bi-file-earmark-excel-fill" aria-hidden="true"></i>
            <span>Export</span>
          </button>` : ''}
        </div>
        <div class="vis-accordion-body" style="display:none;"></div>
      </div>
    </div>
  `);

  $msgRow.after($accordions);

  // Store preview rows keyed on wrapper — live chart/table render uses these (no API call)
  window._visStore = window._visStore || new WeakMap();
  window._visStore.set($accordions[0], { cols, col_labels: col_labels || cols, rows, total_rows, message_id, chart_meta: chart_meta || null });
}

/* ── Number helpers ── */
function _fmtNum(v) {
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (abs >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (abs >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}
function _fmtNumFull(v) {
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
}
/* Format a table cell value: floats → 2 decimal places with thousand separators; integers → locale string; strings unchanged */
function _fmtCell(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') {
    return Number.isInteger(v)
      ? v.toLocaleString()
      : v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return String(v);
}

/* ── Chart view state (offcanvas toggles) ── */
let _chartType = 'bar';        // 'bar' | 'line'
let _chartSort = 'desc';       // 'desc' | 'asc'
const CHART_MAX_POINTS = 24;   // line chart shows at most this many points

/* ── Grouped pivot renderer ────────────────────────────────────────────────────
   Called by _renderChart() when chartMeta.group_col is present.
   Pivots flat rows (time × group × metric) into one ECharts series per group.
   Does NOT touch any other code path. ── */
function _renderGroupedChart(container, cols, rows, xColIdx, yColIdx, groupColIdx, chartMeta, isOffcanvas) {
  const _cn = v => (v === null || v === undefined || v === '') ? 0 : (Number(v) || 0);

  // ── 1. Collect ordered x-axis values ────────────────────────────────────────
  const xOrdered = [];
  const xSeen = new Set();
  rows.forEach(r => {
    const x = String(r[xColIdx] ?? '');
    if (x && x !== 'null' && !xSeen.has(x)) { xSeen.add(x); xOrdered.push(x); }
  });

  // Sort chronologically when the x-axis looks like dates / periods
  const isTimeAxis = xOrdered.length > 0 && xOrdered.every(l =>
    /^\d{4}[-/]/.test(l) ||
    /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i.test(l) ||
    /^Q[1-4]/i.test(l) || /^FY\d/i.test(l)
  );
  if (isTimeAxis) xOrdered.sort();

  // ── 2. Rank groups by total metric → keep top N ─────────────────────────────
  const topN = Math.max(1, (chartMeta && chartMeta.top_n) || 5);
  const totals = {};
  rows.forEach(r => {
    const g = String(r[groupColIdx] ?? '');
    if (!g || g === 'null') return;
    totals[g] = (totals[g] || 0) + _cn(r[yColIdx]);
  });
  const topGroups = Object.entries(totals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(e => e[0]);

  if (!topGroups.length) {
    container.innerHTML = '<p class="vis-error">No chart data available.</p>';
    return;
  }

  // ── 3. Build lookup: group → { xVal → yVal } ────────────────────────────────
  const lookup = {};
  topGroups.forEach(g => { lookup[g] = {}; });
  rows.forEach(r => {
    const g = String(r[groupColIdx] ?? '');
    const x = String(r[xColIdx] ?? '');
    if (lookup[g] !== undefined && x && x !== 'null') {
      lookup[g][x] = _cn(r[yColIdx]);
    }
  });
  const pivotSeries = topGroups.map(g => ({
    name: g,
    data: xOrdered.map(x => (lookup[g][x] !== undefined ? lookup[g][x] : null)),
  }));

  // ── 4. Controls (Bar | Line toggle, offcanvas only) ─────────────────────────
  const parent = container.parentElement;
  if (parent) {
    const ob = parent.querySelector('.chart-controls-bar'); if (ob) ob.remove();
    const op = parent.querySelector('.chart-col-picker');   if (op) op.remove();
  }
  if (parent && isOffcanvas) {
    const rerender = () => {
      container.innerHTML = '';
      if (window._chartOffcanvasInstance) {
        try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
        window._chartOffcanvasInstance = null;
      }
      _renderGroupedChart(container, cols, rows, xColIdx, yColIdx, groupColIdx, chartMeta, isOffcanvas);
    };
    const bar = document.createElement('div');
    bar.className = 'chart-controls-bar';
    const tg = document.createElement('div');
    tg.className = 'chart-seg';
    [['bar', 'Bar'], ['line', 'Line']].forEach(([val, text]) => {
      const b = document.createElement('button');
      b.className = 'chart-seg-btn' + (_chartType === val ? ' active' : '');
      b.textContent = text;
      b.onclick = () => { if (_chartType !== val) { _chartType = val; rerender(); } };
      tg.appendChild(b);
    });
    bar.appendChild(tg);
    parent.insertBefore(bar, container);
  }

  // ── 5. ECharts option ───────────────────────────────────────────────────────
  // Time-series x-axis → default line; user toggle overrides.
  const useBar = (_chartType === 'bar') && !isTimeAxis && !(chartMeta && chartMeta.chart_type === 'line');
  container.style.height = isOffcanvas ? '480px' : '360px';

  const chart = echarts.init(container, null, { renderer: 'canvas' });
  if (isOffcanvas) {
    if (window._chartOffcanvasInstance && window._chartOffcanvasInstance !== chart) {
      try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
    }
    window._chartOffcanvasInstance = chart;
  }

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor   = isDark ? '#94a3b8' : '#64748b';
  const splitColor  = isDark ? 'rgba(255,255,255,.06)' : '#edf0f4';
  const tooltipBg   = isDark ? '#1e293b' : '#ffffff';
  const tooltipBdr  = isDark ? '#334155' : '#e2e8f0';
  const tooltipText = isDark ? '#e2e8f0' : '#0f172a';
  const PAL = ['#10a37f','#6366f1','#f59e0b','#ef4444','#0ea5e9','#a855f7','#14b8a6','#ec4899'];

  const option = {
    backgroundColor: 'transparent',
    animation: true,
    animationDuration: 600,
    animationEasing: 'cubicOut',

    tooltip: {
      trigger: 'axis',
      axisPointer: { type: useBar ? 'shadow' : 'line', lineStyle: { color: '#10a37f', type: 'dashed', width: 1.5 } },
      backgroundColor: tooltipBg,
      borderColor: tooltipBdr,
      borderWidth: 1,
      padding: [10, 14],
      textStyle: { color: tooltipText, fontSize: 13 },
      extraCssText: 'box-shadow:0 4px 16px rgba(0,0,0,.12);border-radius:8px;',
      formatter(params) {
        const arr = Array.isArray(params) ? params : [params];
        const title = `<div style="font-weight:700;margin-bottom:5px;font-size:13px">${arr[0].name}</div>`;
        return title + arr
          .filter(p => p.value !== null && p.value !== undefined)
          .map(p => `<div style="font-size:12.5px">${p.marker}${p.seriesName}: <span style="color:${p.color};font-weight:600">${_fmtNumFull(p.value)}</span></div>`)
          .join('');
      },
    },

    legend: {
      data: topGroups,
      type: 'scroll',
      top: 6,
      icon: 'roundRect',
      itemWidth: 14, itemHeight: 8, itemGap: 12,
      textStyle: { color: textColor, fontSize: 11 },
    },

    grid: { left: '3%', right: '4%', top: '18%', bottom: '14%', containLabel: true },

    xAxis: {
      type: 'category',
      data: xOrdered,
      axisLabel: { rotate: 30, color: textColor, fontSize: 11, margin: 10 },
      axisLine: { lineStyle: { color: splitColor } },
      axisTick: { show: false },
      splitLine: { show: false },
    },

    yAxis: {
      type: 'value',
      axisLabel: { formatter: v => _fmtNum(v), color: textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: splitColor, type: 'dashed' } },
      axisLine: { show: false },
      axisTick: { show: false },
    },

    series: pivotSeries.map((s, si) => {
      const color = PAL[si % PAL.length];
      if (useBar) {
        return {
          name: s.name, type: 'bar',
          data: s.data,
          itemStyle: { color, borderRadius: [3, 3, 0, 0] },
          emphasis: { itemStyle: { opacity: 1, shadowBlur: 8, shadowColor: color + '80' } },
        };
      }
      return {
        name: s.name, type: 'line',
        data: s.data,
        smooth: 0.3,
        symbol: 'circle', symbolSize: 6,
        connectNulls: true,
        lineStyle: { color, width: 2.5 },
        itemStyle: { color, borderWidth: 2, borderColor: isDark ? '#1e293b' : '#fff' },
        emphasis: { scale: true, focus: 'series', itemStyle: { shadowBlur: 10, shadowColor: color + '80' } },
      };
    }),
  };

  chart.setOption(option);
  window.addEventListener('resize', () => { try { chart.resize(); } catch (_) {} });
}

/* ── Chart renderer (ECharts) ── */
// chartMeta (optional): {chart_type, x_col, y_col, top_n, sort} from the backend LLM.
// When present it drives axis/type selection; user toggle controls can still override type+sort.
function _renderChart(container, cols, rows, selectedValIdx, chartMeta) {
  if (typeof echarts === 'undefined') {
    container.innerHTML = '<p class="vis-error">Chart library not loaded.</p>';
    return;
  }
  if (!rows || !rows.length) {
    container.innerHTML = '<p class="vis-error">No data to chart.</p>';
    return;
  }

  const isOffcanvas = container.id === 'chartOffcanvasContainer';
  const firstRow = rows[0];
  const _coerceNum = v => (v === null || v === undefined || v === '') ? NaN : Number(v);
  // Columns that carry aggregate metrics — must not be picked as the category/label axis
  const metricRegex = /revenue|quantity|volume|growth|pct|amount|count|sales|total|invoice/i;

  // ── Label col (dimension) ──────────────────────────────────────────────────────
  // If chartMeta specifies x_col and it exists in cols, use it directly.
  // Otherwise three-pass heuristic: named keyword → first non-metric string → first non-metric col.
  let labelIdx = -1;
  if (chartMeta && chartMeta.x_col) {
    labelIdx = cols.findIndex(c => c.toLowerCase() === chartMeta.x_col.toLowerCase());
  }
  if (labelIdx < 0) labelIdx = cols.findIndex(c =>
    !metricRegex.test(c) &&
    /name|brand|product|zone|territory|period|month|depo|cname|wgbez|arktx|szone/i.test(c)
  );
  if (labelIdx < 0) labelIdx = cols.findIndex((c, i) => !metricRegex.test(c) && typeof firstRow[i] === 'string');
  if (labelIdx < 0) labelIdx = cols.findIndex(c => !metricRegex.test(c));
  if (labelIdx < 0) labelIdx = 0;

  // Columns that are identifiers/codes — numeric but not meaningful metrics
  const codeRegex = /code$|_code|^id$|_id$|^id_|_key$|^key$|kunrg|gsber|^rank$/i;

  // ── Metric cols (all value candidates, excluding label and id/code cols) ──────
  const metricIdxs = cols
    .map((c, i) => i)
    .filter(i => i !== labelIdx && !codeRegex.test(cols[i]) && (metricRegex.test(cols[i]) || !isNaN(_coerceNum(firstRow[i]))));

  // Pick active value col: honour selectedValIdx (user pick) first, then chartMeta.y_col, then first metric.
  let _chartMetaValIdx = -1;
  if (chartMeta && chartMeta.y_col) {
    const mi = cols.findIndex(c => c.toLowerCase() === chartMeta.y_col.toLowerCase());
    if (mi >= 0 && metricIdxs.includes(mi)) _chartMetaValIdx = mi;
  }
  let valIdx = (selectedValIdx != null && selectedValIdx >= 0 && metricIdxs.includes(selectedValIdx))
    ? selectedValIdx
    : (_chartMetaValIdx >= 0 ? _chartMetaValIdx : (metricIdxs[0] ?? (labelIdx === 0 ? 1 : 0)));

  // ── Grouped pivot path (2-dimension data: time × brand, month × zone, etc.) ──
  // Priority 1: LLM emitted group_col in chartMeta.
  // Priority 2: auto-detect — when the x-axis is time-based AND a second
  //   non-metric string column exists, it must be a grouping dimension.
  //   (Safe: only fires on time x-axis, preventing false pivots on simple ranking queries.)
  {
    let _gci = -1;

    // Priority 1: explicit LLM signal
    if (chartMeta && chartMeta.group_col) {
      const gi = cols.findIndex(c => c.toLowerCase() === chartMeta.group_col.toLowerCase());
      if (gi >= 0 && gi !== labelIdx && gi !== valIdx) _gci = gi;
    }

    // Priority 2: auto-detect (only when x-axis labels look like dates/periods)
    if (_gci < 0) {
      const _lbls = rows.map(r => String(r[labelIdx] ?? ''));
      const _isTimeAxis = _lbls.length > 0 && _lbls.every(l =>
        /^\d{4}[-/]/.test(l) ||
        /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i.test(l) ||
        /^Q[1-4]/i.test(l) || /^FY\d/i.test(l)
      );
      if (_isTimeAxis) {
        _gci = cols.findIndex((c, i) =>
          i !== labelIdx &&
          !metricRegex.test(c) &&
          !codeRegex.test(c) &&
          typeof firstRow[i] === 'string' &&
          !!firstRow[i] && firstRow[i] !== 'null'
        );
      }
    }

    if (_gci >= 0) {
      _renderGroupedChart(container, cols, rows, labelIdx, valIdx, _gci, chartMeta, isOffcanvas);
      return;
    }
  }

  // ── Chart shape: time? line? multi-line? ──────────────────────────────────────
  // Time-series data is always a line; chartMeta.chart_type==='line' also forces it;
  // otherwise honour the Bar/Line user toggle.
  const rawLabels = rows.map(r => String(r[labelIdx] ?? ''));
  const isTime = rawLabels.every(l =>
    /^\d{4}[-/]/.test(l) ||                                              // 2024-04, 2024/04
    /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i.test(l) ||    // April 2025, Jan 2025
    /^Q[1-4]\s*\d{4}$/i.test(l) ||                                       // Q1 2025
    /^FY\d/i.test(l)                                                      // FY2025
  );
  const metaWantsLine = !!(chartMeta && chartMeta.chart_type === 'line');
  const asLine = isTime || metaWantsLine || _chartType === 'line';
  // When several metric columns exist (e.g. PrevRev + CurrRev), a line chart
  // overlays them as one line each instead of forcing the user to switch.
  const seriesCols = (asLine && metricIdxs.length >= 2) ? metricIdxs : [valIdx];
  const multiLine = asLine && seriesCols.length > 1;
  // Sequential label (period/month/…) → keep query order; else sort by value.
  const isSequential = isTime || /period|month|date|time|day|week|quarter|year|fiscal/i.test(cols[labelIdx] || '');

  // ── Chart-type + sort controls (offcanvas only) ───────────────────────────────
  const parent = container.parentElement;
  if (parent) {
    const oldBar = parent.querySelector('.chart-controls-bar'); if (oldBar) oldBar.remove();
    const p = parent.querySelector('.chart-col-picker'); if (p) p.remove();
  }
  if (parent && isOffcanvas) {
    const rerender = () => {
      container.innerHTML = '';
      if (window._chartOffcanvasInstance) {
        try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
        window._chartOffcanvasInstance = null;
      }
      _renderChart(container, cols, rows, valIdx, chartMeta);
    };
    const bar = document.createElement('div');
    bar.className = 'chart-controls-bar';

    // Chart type: Bar | Line
    const typeGroup = document.createElement('div');
    typeGroup.className = 'chart-seg';
    [['bar', 'Bar'], ['line', 'Line']].forEach(([val, text]) => {
      const b = document.createElement('button');
      b.className = 'chart-seg-btn' + (_chartType === val ? ' active' : '');
      b.textContent = text;
      b.onclick = () => { if (_chartType !== val) { _chartType = val; rerender(); } };
      typeGroup.appendChild(b);
    });

    // Sort: Desc | Asc
    const sortGroup = document.createElement('div');
    sortGroup.className = 'chart-seg';
    [['desc', 'Desc'], ['asc', 'Asc']].forEach(([val, text]) => {
      const b = document.createElement('button');
      b.className = 'chart-seg-btn' + (_chartSort === val ? ' active' : '');
      b.textContent = text;
      b.onclick = () => { if (_chartSort !== val) { _chartSort = val; rerender(); } };
      sortGroup.appendChild(b);
    });

    bar.appendChild(typeGroup);
    bar.appendChild(sortGroup);
    parent.insertBefore(bar, container);
  }

  // ── Column picker (single-metric selection; hidden when lines are overlaid) ────
  if (metricIdxs.length > 1 && parent && isOffcanvas && !multiLine) {
    const picker = document.createElement('div');
    picker.className = 'chart-col-picker';
    metricIdxs.forEach(idx => {
      const btn = document.createElement('button');
      btn.className = 'chart-col-btn' + (idx === valIdx ? ' active' : '');
      btn.textContent = cols[idx];
      btn.onclick = () => {
        container.innerHTML = '';
        if (window._chartOffcanvasInstance) {
          try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
          window._chartOffcanvasInstance = null;
        }
        _renderChart(container, cols, rows, idx, chartMeta);
      };
      picker.appendChild(btn);
    });
    parent.insertBefore(picker, container);
  }

  // ── Data preparation ──────────────────────────────────────────────────────────
  // One record per row: its label plus the value of every series column.
  let recs = rows.map(r => ({
    label: String(r[labelIdx] ?? ''),
    vals: seriesCols.map(ci => _coerceNum(r[ci])),
  })).filter(d => d.label !== '' && d.label !== 'null');
  // Single-series: drop rows whose only value is missing (matches old behaviour).
  if (!multiLine) recs = recs.filter(d => !isNaN(d.vals[0]));

  // Order: a sequential line (months/periods) stays in query order so the axis
  // reads chronologically; everything else sorts by the primary metric.
  if (!(isSequential && asLine)) {
    const dir = _chartSort === 'asc' ? 1 : -1;
    recs.sort((a, b) => {
      const av = a.vals[0], bv = b.vals[0];
      if (isNaN(av)) return 1;
      if (isNaN(bv)) return -1;
      return dir * (av - bv);
    });
  }
  // Line view is capped to keep the axis readable.
  if (asLine && recs.length > CHART_MAX_POINTS) recs = recs.slice(0, CHART_MAX_POINTS);

  const labels = recs.map(d => d.label);
  const seriesValues = seriesCols.map((_, si) => recs.map(d => d.vals[si]));
  const values = seriesValues[0] || [];   // primary series (bar + single line)

  // Series that are far smaller in magnitude go on a secondary y-axis so a
  // percentage/growth line doesn't get flattened against a revenue line.
  const seriesMax = seriesValues.map(vals =>
    Math.max(1, ...vals.map(v => (isNaN(v) ? 0 : Math.abs(v)))));
  const globalMax = Math.max(...seriesMax);
  const yIndexFor = si =>
    (multiLine && globalMax / seriesMax[si] > 30) ? 1 : 0;
  const hasSecondaryAxis = multiLine && seriesCols.some((_, si) => yIndexFor(si) === 1);
  const SERIES_PALETTE = ['#10a37f', '#6366f1', '#f59e0b', '#ef4444', '#0ea5e9', '#a855f7', '#14b8a6', '#ec4899'];

  // ── Chart init ────────────────────────────────────────────────────────────────
  if (isOffcanvas) {
    container.style.height = asLine ? '440px' : (Math.max(400, values.length * 36 + 80) + 'px');
  } else {
    container.style.height = '360px';
  }

  const chart = echarts.init(container, null, { renderer: 'canvas' });

  if (isOffcanvas) {
    if (window._chartOffcanvasInstance && window._chartOffcanvasInstance !== chart) {
      try { window._chartOffcanvasInstance.dispose(); } catch (_) {}
    }
    window._chartOffcanvasInstance = chart;
  }

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor    = isDark ? '#94a3b8' : '#64748b';
  const splitColor   = isDark ? 'rgba(255,255,255,.06)' : '#edf0f4';
  const tooltipBg    = isDark ? '#1e293b' : '#ffffff';
  const tooltipBdr   = isDark ? '#334155' : '#e2e8f0';
  const tooltipText  = isDark ? '#e2e8f0' : '#0f172a';
  const maxVal       = Math.max(...values) || 1;

  const barColor = (v) => ({
    color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
      { offset: 0, color: '#0b8a65' },
      { offset: 1, color: '#34c997' },
    ]),
    borderRadius: [0, 5, 5, 0],
    opacity: 0.5 + 0.5 * (v / maxVal),
  });

  const option = {
    backgroundColor: 'transparent',
    animation: true,
    animationDuration: 650,
    animationEasing: 'cubicOut',

    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: asLine ? 'line' : 'shadow',
        shadowStyle: { color: 'rgba(16,163,127,.08)' },
        lineStyle: { color: '#10a37f', type: 'dashed', width: 1.5 },
      },
      backgroundColor: tooltipBg,
      borderColor: tooltipBdr,
      borderWidth: 1,
      padding: [10, 14],
      textStyle: { color: tooltipText, fontSize: 13 },
      extraCssText: 'box-shadow:0 4px 16px rgba(0,0,0,.12);border-radius:8px;',
      formatter(params) {
        const arr = Array.isArray(params) ? params : [params];
        const title = `<div style="font-weight:700;margin-bottom:5px;font-size:13px">${arr[0].name}</div>`;
        if (multiLine) {
          return title + arr.map(p =>
            `<div style="font-size:12.5px">${p.marker}${p.seriesName}: <span style="color:${p.color};font-weight:600">${_fmtNumFull(p.value)}</span></div>`
          ).join('');
        }
        const p = arr[0];
        const rankLine = asLine ? '' :
          `<div style="margin-top:5px;font-size:11px;color:${textColor}">Rank #${p.dataIndex + 1} of ${rows.length}</div>`;
        return title +
          `<div style="font-size:13px">${cols[valIdx] || 'Value'}: <span style="color:#10a37f;font-weight:600">${_fmtNumFull(p.value)}</span></div>
                ${rankLine}`;
      },
    },

    legend: multiLine ? {
      data: seriesCols.map(ci => cols[ci]),
      top: 6,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 8,
      itemGap: 14,
      textStyle: { color: textColor, fontSize: 11 },
    } : undefined,

    grid: asLine
      ? { left: '3%', right: hasSecondaryAxis ? '10%' : '5%', top: multiLine ? '16%' : '8%', bottom: '14%', containLabel: true }
      : { left: '2%', right: '17%', top: '2%', bottom: '2%', containLabel: true },

    xAxis: asLine ? {
      type: 'category', data: labels,
      axisLabel: { rotate: 30, color: textColor, fontSize: 11, margin: 10 },
      axisLine: { lineStyle: { color: splitColor } },
      axisTick: { show: false },
      splitLine: { show: false },
    } : {
      type: 'value',
      axisLabel: { formatter: v => _fmtNum(v), color: textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: splitColor, type: 'dashed' } },
      axisLine: { show: false },
      axisTick: { show: false },
    },

    yAxis: asLine ? (hasSecondaryAxis ? [{
      type: 'value',
      axisLabel: { formatter: v => _fmtNum(v), color: textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: splitColor, type: 'dashed' } },
      axisLine: { show: false },
      axisTick: { show: false },
    }, {
      type: 'value',
      axisLabel: { formatter: v => _fmtNum(v), color: textColor, fontSize: 11 },
      splitLine: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
    }] : {
      type: 'value',
      axisLabel: { formatter: v => _fmtNum(v), color: textColor, fontSize: 11 },
      splitLine: { lineStyle: { color: splitColor, type: 'dashed' } },
      axisLine: { show: false },
      axisTick: { show: false },
    }) : {
      type: 'category',
      data: labels,
      inverse: true,
      axisLabel: { width: 170, overflow: 'truncate', color: textColor, fontSize: 11, tooltip: { show: true } },
      axisLine: { show: false },
      axisTick: { show: false },
    },

    series: asLine ? seriesCols.map((ci, si) => {
      const color = SERIES_PALETTE[si % SERIES_PALETTE.length];
      return {
        name: cols[ci],
        type: 'line',
        data: seriesValues[si],
        yAxisIndex: yIndexFor(si),
        smooth: 0.4,
        symbol: 'circle',
        symbolSize: 7,
        connectNulls: true,
        lineStyle: { color, width: 2.5 },
        itemStyle: { color, borderWidth: 2.5, borderColor: isDark ? '#1e293b' : '#fff' },
        // Area fill only for a single line — overlapping fills look muddy.
        areaStyle: multiLine ? undefined : {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(16,163,127,.28)' },
            { offset: 1, color: 'rgba(16,163,127,.02)' },
          ]),
        },
        emphasis: { scale: true, focus: multiLine ? 'series' : 'none', itemStyle: { shadowBlur: 10, shadowColor: 'rgba(16,163,127,.45)' } },
      };
    }) : [{
      type: 'bar',
      data: values.map(v => ({ value: v, itemStyle: barColor(v) })),
      barMaxWidth: 28,
      label: {
        show: true,
        position: 'right',
        formatter: p => _fmtNum(p.value),
        color: textColor,
        fontSize: 11,
        fontWeight: 500,
      },
      emphasis: {
        itemStyle: { opacity: 1, shadowBlur: 12, shadowColor: 'rgba(16,163,127,.45)' },
        label: { color: isDark ? '#e2e8f0' : '#0f172a', fontWeight: 700 },
      },
    }],
  };

  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}

/* ── Chart from API (historical messages — re-runs stored KQL with limit 100) ── */
async function _loadChartFromApi(container, messageId) {
  const token = localStorage.getItem('auth_token');
  if (!token) return;

  const $c = $(container);
  if (container.id !== 'chartOffcanvasContainer') {
    $c.css('height', '360px');
  }
  $c.html('<div class="vis-loading"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading chart…</div>');

  try {
    const params = new URLSearchParams({ message_id: messageId, mode: 'chart', page: 1, page_size: 100 });
    const resp = await fetch(`${apiBase}/sales/data/?${params}`, {
      headers: { Authorization: 'Bearer ' + token },
    });
    if (!resp.ok) {
      $c.css('height', '').html('<p class="vis-error">Failed to load chart data.</p>');
      return;
    }
    const data = await resp.json();
    $c.html('');
    _renderChart(container, data.cols, data.rows);
  } catch {
    $c.css('height', '').html('<p class="vis-error">Error loading chart data.</p>');
  }
}

/* ── Preview table (from SSE rows) ── */
function _renderPreviewTable(container, cols, rows, totalRows, messageId) {
  const isTruncated = totalRows > rows.length;

  // Sort descending by the most relevant numeric column
  // Priority: revenue/sales/amount → volume/quantity/count → first float column (avoids integer IDs)
  let numColIdx = cols.findIndex(c => /revenue|sales|amount/i.test(c));
  if (numColIdx < 0) numColIdx = cols.findIndex(c => /volume|quantity|count/i.test(c));
  if (numColIdx < 0) numColIdx = cols.findIndex((c, i) =>
    rows.length > 0 && typeof rows[0][i] === 'number' && !Number.isInteger(rows[0][i])
  );
  const sortedRows = numColIdx >= 0
    ? [...rows].sort((a, b) => Number(b[numColIdx] ?? 0) - Number(a[numColIdx] ?? 0))
    : rows;

  const thead = `<thead><tr>${cols.map(c => `<th>${escapeAttr(String(c))}</th>`).join('')}</tr></thead>`;
  const buildTbody = (data) =>
    `<tbody>${data.map(row =>
      `<tr>${row.map(v => `<td>${escapeAttr(_fmtCell(v))}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;

  const $wrap = $(`
    <div class="vis-table-wrap">
      <div class="vis-table-toolbar">
        <input class="vis-search-input" placeholder="Search preview…" aria-label="Search table" />
        <span class="vis-row-count">Showing ${sortedRows.length} of ${totalRows} records</span>
        ${isTruncated && messageId
          ? `<button class="vis-load-all-btn" data-message-id="${messageId}">Load all records</button>`
          : ''}
      </div>
      <div class="vis-table-scroll">
        <table class="vis-data-table">${thead}${buildTbody(sortedRows)}</table>
      </div>
    </div>
  `);

  // Client-side search on preview rows
  $wrap.find('.vis-search-input').on('input', function () {
    const q = $(this).val().toLowerCase();
    $wrap.find('.vis-data-table tbody tr').each(function () {
      $(this).toggle(!q || $(this).text().toLowerCase().includes(q));
    });
  });

  // Load all records
  $wrap.find('.vis-load-all-btn').on('click', function () {
    const mid = $(this).data('message-id');
    $(container).empty();
    _loadFullTable(container, mid, 1, '');
  });

  $(container).empty().append($wrap);
}

/* ── Full paginated table — jQuery DataTables server-side ── */
async function _loadFullTable(container, messageId) {
  const token = localStorage.getItem('auth_token');
  if (!token) return;

  const $c = $(container);

  // Destroy any existing DataTable instance before re-initialising
  const $prev = $c.find('table');
  if ($prev.length && $.fn.DataTable && $.fn.DataTable.isDataTable($prev[0])) {
    $prev.DataTable().destroy(true);
  }

  $c.html('<div class="vis-loading"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading…</div>');

  // Initial fetch: discover column names + first page without a round-trip penalty
  let prefetch;
  try {
    const r = await fetch(
      `${apiBase}/sales/data/?${new URLSearchParams({ message_id: messageId, page: 1, page_size: 50 })}`,
      { headers: { Authorization: 'Bearer ' + token } }
    );
    if (!r.ok) { $c.html('<p class="vis-error">Failed to load data.</p>'); return; }
    prefetch = await r.json();
  } catch {
    $c.html('<p class="vis-error">Error loading data.</p>');
    return;
  }

  const { cols, col_labels, rows: prefetchRows, total_rows } = prefetch;
  const displayCols = col_labels || cols;

  // Update the accordion badge with the real total (API count, not the SSE preview cap)
  const $badge = $c.closest('.vis-accordion').find('.vis-accordion-count');
  if ($badge.length && total_rows != null) {
    $badge.text(Number(total_rows).toLocaleString() + ' records');
  }

  $c.html(`
    <div class="vis-dt-wrap">
      <table class="vis-data-table display">
        <thead><tr>${displayCols.map(c => `<th>${escapeAttr(String(c))}</th>`).join('')}</tr></thead>
        <tbody></tbody>
      </table>
    </div>
  `);

  let prefetchConsumed = false;

  $c.find('table').DataTable({
    serverSide: true,
    processing: true,
    pageLength: 50,
    lengthMenu: [[25, 50, 100], [25, 50, 100]],
    order: [],   // initial order comes from the KQL's preserved ORDER BY
    scrollX: true,

    columns: displayCols.map(c => ({ title: escapeAttr(String(c)) })),

    ajax(dtParams, callback) {
      // First call: serve the prefetched data to avoid a redundant API round-trip
      if (!prefetchConsumed && dtParams.start === 0 && !dtParams.search.value) {
        prefetchConsumed = true;
        callback({
          draw: dtParams.draw,
          recordsTotal: total_rows || 0,
          recordsFiltered: total_rows || 0,
          data: prefetchRows.map(row => row.map(v => _fmtCell(v))),
        });
        return;
      }

      const orderIdx = dtParams.order[0]?.column;
      const sortCol  = orderIdx != null ? (cols[orderIdx] || '') : '';  // raw name for ADX sort
      const sortDir  = dtParams.order[0]?.dir || 'desc';
      const page     = Math.floor(dtParams.start / dtParams.length) + 1;

      const params = new URLSearchParams({
        message_id: messageId,
        page,
        page_size: dtParams.length,
      });
      if (dtParams.search.value) params.set('search', dtParams.search.value);
      if (sortCol) { params.set('sort_col', sortCol); params.set('sort_dir', sortDir); }

      fetch(`${apiBase}/sales/data/?${params}`, { headers: { Authorization: 'Bearer ' + token } })
        .then(r => r.json())
        .then(result => callback({
          draw: dtParams.draw,
          recordsTotal: result.total_rows || total_rows || 0,
          recordsFiltered: result.total_rows || total_rows || 0,
          data: (result.rows || []).map(row => row.map(v => _fmtCell(v))),
        }))
        .catch(() => callback({ draw: dtParams.draw, recordsTotal: 0, recordsFiltered: 0, data: [] }));
    },

    language: {
      processing: '<div class="vis-loading"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading…</div>',
      paginate: { first: '«', last: '»', previous: '‹', next: '›' },
      search: '',
      searchPlaceholder: 'Search records…',
      info: 'Showing _START_–_END_ of _TOTAL_ records',
      infoEmpty: 'No records found',
      infoFiltered: '(filtered from _MAX_ total)',
      lengthMenu: 'Show _MENU_ rows',
    },

    dom: '<"vis-dt-top"<"vis-dt-length"l><"vis-dt-search"f>>t<"vis-dt-bottom"<"vis-dt-info"i><"vis-dt-pages"p>>',
  });
}

/* ── Global: dropdown & misc handlers ── */

function toggleChatDropdown(event, chatId) {
  event.stopPropagation();
  $('.chat-dropdown').not(`#dropdown-${chatId}`).removeClass('show');
  $('.user-dropdown').removeClass('show');
  $(`#dropdown-${chatId}`).toggleClass('show');
}

function shareChat(chatId) {
  alert(`Share link for chat ${chatId}`);
  $('.chat-dropdown').removeClass('show');
}

function renameChat(chatId) {
  $('.chat-dropdown').removeClass('show');
  const $item  = $(`.chat-item[data-chat-id="${chatId}"]`);
  const $title = $item.find('.chat-title');
  const current = $title.text();

  const $input = $(`<input type="text" class="chat-rename-input"
      value="${current.replace(/"/g, '&quot;')}"
      aria-label="Rename chat" />`);
  $title.replaceWith($input);
  $input.focus().select();

  function saveRename() {
    const newTitle = $input.val().trim() || current;
    $input.replaceWith(`<div class="chat-title">${newTitle}</div>`);
    if (newTitle === current) return;
    const token = localStorage.getItem('auth_token');
    $.ajax({
      url: apiBase + `/conversations/${chatId}/`,
      method: 'PUT',
      headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
      data: JSON.stringify({ title: newTitle }),
    });
  }

  $input.on('keydown', function (e) {
    if (e.key === 'Enter')  { e.preventDefault(); saveRename(); }
    if (e.key === 'Escape') { $input.replaceWith(`<div class="chat-title">${current}</div>`); }
  });
  $input.on('blur', saveRename);
}

function archiveChat(chatId) {
  if (confirm('Archive this chat?')) {
    $(`.chat-item[data-chat-id="${chatId}"]`).fadeOut(300, function () { $(this).remove(); });
  }
  $('.chat-dropdown').removeClass('show');
}

function deleteChat(chatId) {
  if (confirm('Delete this chat?')) {
    $(`.chat-item[data-chat-id="${chatId}"]`).fadeOut(300, function () { $(this).remove(); });
  }
  $('.chat-dropdown').removeClass('show');
}

function toggleUserDropdown() {
  $('.chat-dropdown').removeClass('show');
  $('#user-dropdown').toggleClass('show');
}

function logOut() {
  const token        = localStorage.getItem('auth_token');
  const refreshToken = localStorage.getItem('refresh_token');

  $.ajax({
    url: apiBase + '/user/logout/',
    method: 'POST',
    headers: { Authorization: 'Bearer ' + token },
    data: JSON.stringify({ refresh_token: refreshToken }),
    contentType: 'application/json',
    success() {
      localStorage.clear();
      window.location.href = '/welcome';
    },
    error(xhr) {
      let detail;
      try { detail = JSON.parse(xhr.responseText).detail; } catch { detail = xhr.responseText; }
      if (detail && detail.includes('Token is blacklisted')) {
        localStorage.clear();
        window.location.href = '/welcome';
      } else {
        Swal.fire({ icon: 'error', title: 'Oops…', text: 'Logout failed. Please try again.' });
      }
    },
  });
}

function openSettings() {
  $('.chat-dropdown, .user-dropdown').removeClass('show');
  const modalEl = document.getElementById('settingsModal');
  if (modalEl) new bootstrap.Modal(modalEl).show();
}

function saveSettings() {
  const theme = $('input[name="theme"]:checked').val() || 'auto';
  applyTheme(theme);
  const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
  if (modal) modal.hide();
}

/* Close dropdowns on outside click */
$(document).on('click', function (e) {
  if (!$(e.target).closest('.chat-item, .user-info').length) {
    $('.chat-dropdown, .user-dropdown').removeClass('show');
  }
});
