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
             <div class="vis-accordion" data-type="chart">
               <button class="vis-accordion-header">
                 <i class="bi bi-bar-chart" aria-hidden="true"></i>
                 <span>Chart</span>
                 <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
               </button>
               <div class="vis-accordion-body" style="display:none;"></div>
             </div>
             <div class="vis-accordion" data-type="table">
               <button class="vis-accordion-header">
                 <i class="bi bi-table" aria-hidden="true"></i>
                 <span>Table</span>
                 <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
               </button>
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
                  rows: payload.rows,
                  total_rows: payload.total_rows || payload.rows.length,
                  message_id: payload.message_id || null,
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

  /* ── Visualization: unified accordion handler (chart + table, live + history) ── */
  $(document).on('click', '.vis-accordion-header', function () {
    const $header = $(this);
    const $accordion = $header.closest('.vis-accordion');
    const $body = $accordion.find('.vis-accordion-body');
    const $wrap = $accordion.closest('.vis-accordions');
    const type = $accordion.data('type');
    const isOpen = $body.is(':visible');

    $body.toggle(!isOpen);
    $header.toggleClass('open', !isOpen);

    if (!isOpen && !$body.data('loaded')) {
      $body.data('loaded', true);

      // Live session: vis-uid; historical message: message-id
      const messageId = $wrap.data('vis-uid') || $wrap.data('message-id');

      window._visStore = window._visStore || new WeakMap();
      const preview = window._visStore.get($wrap[0]);

      if (type === 'chart') {
        if (preview) {
          _renderChart($body[0], preview.cols, preview.rows);
        } else if (messageId) {
          _loadChartFromApi($body[0], messageId);
        } else {
          $body.html('<p class="vis-error">No chart data available.</p>');
        }
      } else if (type === 'table') {
        if (preview && !$wrap.hasClass('vis-history-mode')) {
          _renderPreviewTable($body[0], preview.cols, preview.rows, preview.total_rows, preview.message_id);
        } else if (messageId) {
          _loadFullTable($body[0], messageId, 1, '');
        } else {
          $body.html('<p class="vis-error">No table data available.</p>');
        }
      }
    }
  });
});

/* ═══════════════════════════════════════════════════════
   Visualization helpers (outside $(function) — global scope)
   ═══════════════════════════════════════════════════════ */

/* Attach two accordions (Chart + Table) after a live assistant message row */
function _attachVisToggles($msgRow, chartData) {
  const { cols, rows, total_rows, message_id } = chartData;
  const uid = message_id || ('vis-' + Date.now());

  const $accordions = $(`
    <div class="vis-accordions" data-vis-uid="${uid}">
      <div class="vis-accordion" data-type="chart">
        <button class="vis-accordion-header">
          <i class="bi bi-bar-chart" aria-hidden="true"></i>
          <span>Chart</span>
          <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
        </button>
        <div class="vis-accordion-body" style="display:none;"></div>
      </div>
      <div class="vis-accordion" data-type="table">
        <button class="vis-accordion-header">
          <i class="bi bi-table" aria-hidden="true"></i>
          <span>Table</span>
          <span class="vis-accordion-count">${total_rows} records</span>
          <i class="bi bi-chevron-down vis-chevron" aria-hidden="true"></i>
        </button>
        <div class="vis-accordion-body" style="display:none;"></div>
      </div>
    </div>
  `);

  $msgRow.after($accordions);

  // Store preview rows keyed on wrapper — live chart/table render uses these (no API call)
  window._visStore = window._visStore || new WeakMap();
  window._visStore.set($accordions[0], { cols, rows, total_rows, message_id });
}

/* ── Chart renderer (ECharts) ── */
function _renderChart(container, cols, rows) {
  if (typeof echarts === 'undefined') {
    container.innerHTML = '<p class="vis-error">Chart library not loaded.</p>';
    return;
  }
  if (!rows || !rows.length) {
    container.innerHTML = '<p class="vis-error">No data to chart.</p>';
    return;
  }

  container.style.height = '360px';
  const chart = echarts.init(container);

  const firstRow = rows[0];

  // Detect label column (string) and value column (number)
  let labelIdx = cols.findIndex((c, i) =>
    /name|brand|product|dealer|zone|territory|gsber|period|month|depo/i.test(c) ||
    typeof firstRow[i] === 'string'
  );
  let valIdx = cols.findIndex((c, i) =>
    /revenue|quantity|volume|growth|pct|amount|count|sales/i.test(c) ||
    typeof firstRow[i] === 'number'
  );

  if (labelIdx < 0) labelIdx = 0;
  if (valIdx < 0 || valIdx === labelIdx) valIdx = labelIdx === 0 ? 1 : 0;

  const labels = rows.map(r => String(r[labelIdx] ?? ''));
  const values = rows.map(r => Number(r[valIdx] ?? 0));

  // Use line chart if labels look like dates/periods
  const isTime = labels.every(l => /^\d{4}[-/]/.test(l));

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: isTime ? 'line' : 'shadow' } },
    grid: { left: isTime ? '8%' : '25%', right: '5%', top: '10%', bottom: '12%', containLabel: true },
    xAxis: isTime
      ? { type: 'category', data: labels, axisLabel: { rotate: 30 } }
      : { type: 'value', name: cols[valIdx] || '' },
    yAxis: isTime
      ? { type: 'value', name: cols[valIdx] || '' }
      : { type: 'category', data: labels, axisLabel: { width: 160, overflow: 'truncate' } },
    series: [{
      type: isTime ? 'line' : 'bar',
      data: values,
      barMaxWidth: 32,
      smooth: isTime,
      itemStyle: { color: '#10a37f' },
    }],
  });

  window.addEventListener('resize', () => chart.resize());
}

/* ── Chart from API (historical messages — re-runs stored KQL with limit 100) ── */
async function _loadChartFromApi(container, messageId) {
  const token = localStorage.getItem('auth_token');
  if (!token) return;

  const $c = $(container);
  $c.css('height', '360px').html(
    '<div class="vis-loading"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading chart…</div>'
  );

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

  const thead = `<thead><tr>${cols.map(c => `<th>${escapeAttr(String(c))}</th>`).join('')}</tr></thead>`;
  const buildTbody = (data) =>
    `<tbody>${data.map(row =>
      `<tr>${row.map(v => `<td>${escapeAttr(String(v ?? ''))}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;

  const $wrap = $(`
    <div class="vis-table-wrap">
      <div class="vis-table-toolbar">
        <input class="vis-search-input" placeholder="Search preview…" aria-label="Search table" />
        <span class="vis-row-count">Showing ${rows.length} of ${totalRows} records</span>
        ${isTruncated && messageId
          ? `<button class="vis-load-all-btn" data-message-id="${messageId}">Load all records</button>`
          : ''}
      </div>
      <div class="vis-table-scroll">
        <table class="vis-data-table">${thead}${buildTbody(rows)}</table>
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

/* ── Full paginated table (fetches from /api/sales/data/) ── */
async function _loadFullTable(container, messageId, page, search) {
  const token = localStorage.getItem('auth_token');
  if (!token) return;

  const $c = $(container);
  $c.html('<div class="vis-loading"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading…</div>');

  try {
    const params = new URLSearchParams({ message_id: messageId, page, page_size: 50 });
    if (search) params.set('search', search);

    const resp = await fetch(`${apiBase}/sales/data/?${params}`, {
      headers: { Authorization: 'Bearer ' + token },
    });
    if (!resp.ok) { $c.html('<p class="vis-error">Failed to load data.</p>'); return; }

    const data = await resp.json();
    const { cols, rows, page: pg, page_size, total_rows, total_pages } = data;

    // Cache total_rows on container for pagination without re-fetching count
    if (total_rows !== undefined) $c.data('total_rows', total_rows);
    if (total_pages !== undefined) $c.data('total_pages', total_pages);
    const cachedTotal = $c.data('total_rows') || rows.length;
    const cachedPages = $c.data('total_pages') || 1;

    const thead = `<thead><tr>${cols.map(c => `<th>${escapeAttr(String(c))}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${rows.map(row =>
      `<tr>${row.map(v => `<td>${escapeAttr(String(v ?? ''))}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;

    let searchTimer;
    const $wrap = $(`
      <div class="vis-table-wrap">
        <div class="vis-table-toolbar">
          <input class="vis-search-input vis-server-search" placeholder="Search all records…" value="${escapeAttr(search)}" aria-label="Search all records" />
          <span class="vis-row-count">${cachedTotal} total records</span>
        </div>
        <div class="vis-table-scroll">
          <table class="vis-data-table">${thead}${tbody}</table>
        </div>
        <div class="vis-pagination">
          <button class="vis-page-btn vis-page-prev" ${pg <= 1 ? 'disabled' : ''}>&#8249; Prev</button>
          <span class="vis-page-label">Page ${pg} of ${cachedPages}</span>
          <button class="vis-page-btn vis-page-next" ${pg >= cachedPages ? 'disabled' : ''}>Next &#8250;</button>
        </div>
      </div>
    `);

    $wrap.find('.vis-server-search').on('input', function () {
      clearTimeout(searchTimer);
      const q = $(this).val().trim();
      searchTimer = setTimeout(() => _loadFullTable(container, messageId, 1, q), 300);
    });
    $wrap.find('.vis-page-prev').on('click', () => _loadFullTable(container, messageId, pg - 1, search));
    $wrap.find('.vis-page-next').on('click', () => _loadFullTable(container, messageId, pg + 1, search));

    $c.empty().append($wrap);
  } catch {
    $c.html('<p class="vis-error">Error loading data.</p>');
  }
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
