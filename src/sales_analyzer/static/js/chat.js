'use strict';
 // Mobile menu toggle functionality
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

    // Close sidebar when clicking on a chat item (mobile)
    $('.chats-section').on('click', '.chat-item', function () {
      if (window.innerWidth <= 768) {
        $('#sidebar').removeClass('open');
        $('#sidebar-overlay').removeClass('show');
      }
    });

    // Close sidebar when starting new chat (mobile)
    $('.sidebar-menu .menu-item:contains("New chat")').on('click', function () {
      if (window.innerWidth <= 768) {
        $('#sidebar').removeClass('open');
        $('#sidebar-overlay').removeClass('show');
      }
    });
  });

  // Make onModelSelected global
  window.onModelSelected = function (selector) {
    console.log('Selected:', selector);
  };

  document.addEventListener('DOMContentLoaded', () => {
    const menu = document.getElementById('modelMenu');
    const label = document.getElementById('current-model');
    const toggle = document.getElementById('modelDropdown');

    menu.addEventListener('click', (e) => {
      const item = e.target.closest('.dropdown-item[data-selector]');
      if (!item) return;

      const selector = item.dataset.selector;
      const text = item.textContent.trim();

      label.textContent = text;
      window.onModelSelected(selector);
      bootstrap.Dropdown.getOrCreateInstance(toggle).hide();
    });
  });

  $(function () {
    let isSubmitting = false;
    let typingIndicator = null;

    // ===== Pagination state for the chats list =====
    let chatsPage = 1;
    let chatsHasNext = true;
    let chatsIsLoading = false;

    // ===== Messages pagination state (next PATH approach) =====
    let msgsIsLoading = false;
    let msgsHasNext   = false;     // whether there’s an older page
    let msgsNextPath  = null;      // relative API path for the next (older) page

    // User info (from localStorage)
    let fullName = localStorage.getItem('full_name') || 'User';
    let userName = localStorage.getItem('username') || 'user';
    let designation = localStorage.getItem('designation') || 'Free';

    $('.user-info .user-name').text(fullName);
    $('.user-info .user-plan').text(designation);

    const initials = (fullName.trim()
      ? fullName.trim().split(/\s+/).map((s) => s[0]).join('')
      : userName.slice(0, 2)
    )
      .slice(0, 2)
      .toUpperCase();

    $('.user-info .user-avatar').text(initials || 'U');

    function getAuthToken() {
      return localStorage.getItem('auth_token');
    }

    function refreshToken() {
      $.ajax({
        url: apiBase + '/user/token/refresh/',
        method: 'POST',
        data: JSON.stringify({ refresh: localStorage.getItem('refresh_token') }),
        contentType: 'application/json',
        success(resp) {
          localStorage.setItem('auth_token', resp.access);
          localStorage.setItem('refresh_token', resp.refresh);
          // reload chats from page 1 on refresh
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
      if (!token) {
        window.location.href = '/welcome';
        return;
      }
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
      const $c = $('#chat-content');
      $c.scrollTop($c.prop('scrollHeight'));
    }

    function updateSendButton() {
      const text = $('#message-input').val().trim();
      $('#send-btn').prop('disabled', text.length === 0 || isSubmitting);
    }

    // ----- Chats: Load more button helpers -----
    function renderLoadMoreButton() {
      const $sec = $('.chats-section');
      const id = 'load-more-convos';
      $('#' + id).remove(); // avoid duplicates

      if (chatsHasNext) {
        $sec.append(`
          <div id="${id}" class="menu-item load-more-btn" style="margin-top:6px;">
            <span class="icon">＋</span> Load more
          </div>
        `);
      }
    }

    function setLoadMoreLoading(loading) {
      const $btn = $('#load-more-convos');
      if (!$btn.length) return;
      if (loading) {
        $btn.addClass('disabled').css('pointer-events', 'none').text('Loading…');
      } else {
        $btn.removeClass('disabled').css('pointer-events', '').html('<span class="icon">＋</span> Load more');
      }
    }

    /**
     * Fetch conversations (server paginated)
     * @param {boolean} append - append results (true) or reset (false)
     * @param {function} cb    - optional callback after render
     */
    function fetchChats(append = false, cb) {
      if (chatsIsLoading) return;

      const $sec = $('.chats-section');

      if (!append) {
        chatsPage = 1;
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

      const url = `/conversations/?page=${chatsPage}`;
      sendAuthenticatedRequest(
        url,
        'GET',
        null,
        (resp) => {
          chatsIsLoading = false;
          setLoadMoreLoading(false);

          // Normalize DRF-style pagination (count/next/previous/results)
          const meta =
            resp && (resp.meta
              ? resp.meta
              : typeof resp === 'object' && ('next' in resp || 'previous' in resp || 'count' in resp)
              ? { next: resp.next, previous: resp.previous, count: resp.count }
              : null);

          const list = resp && resp.results ? resp.results : Array.isArray(resp) ? resp : [];

          if (!append && list.length === 0) {
            $sec.append('<div class="no-conversations">No previous conversations</div>');
          } else {
            list.forEach((c) => {
              const active = c.uuid === currentChatId ? 'active' : '';
              if ($sec.find(`.chat-item[data-chat-id="${c.uuid}"]`).length) return; // de-dupe
              $sec.append(`
                <div class="chat-item ${active}" data-chat-id="${c.uuid}">
                  <div class="chat-title">${c.title || 'Untitled Chat'}</div>
                  <button class="chat-menu-btn" onclick="toggleChatDropdown(event,'${c.uuid}')">…</button>
                  <div class="chat-dropdown" id="dropdown-${c.uuid}">
                    <button class="dropdown-item" onclick="shareChat('${c.uuid}')">Share</button>
                    <button class="dropdown-item" onclick="renameChat('${c.uuid}')">Rename</button>
                    <button class="dropdown-item" onclick="archiveChat('${c.uuid}')">Archive</button>
                    <button class="dropdown-item delete" onclick="deleteChat('${c.uuid}')">Delete</button>
                  </div>
                </div>
              `);
            });
          }

          // Pagination state (show "Load more" only when next exists)
          if (meta) {
            const hasNext = !!meta.next;
            chatsHasNext = hasNext;

            if (hasNext) {
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

    // ===== Messages: utilities =====
    function ensureMessageShell() {
      const $chat = $('#chat-content');
      if (!$('#messages-list').length) {
        $chat.html(`
          <div id="load-prev-wrap" class="text-center"></div>
          <div id="messages-list"></div>
        `);
      }
    }

    // Initial (first page) loader
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

    // Load previous button helpers
    function renderLoadPrevButton() {
      const $wrap = $('#load-prev-wrap');
      if (!$wrap.length) return;

      if (!msgsHasNext) {
        $wrap.empty();
        return;
      }

      if (!$('#load-prev-msgs').length) {
        $wrap.html(`
          <button id="load-prev-msgs"
                  class="btn btn-outline-secondary btn-sm d-none"
                  style="margin: 6px auto 12px; display:inline-flex; align-items:center; gap:.4rem;">
            <span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span>
            Load previous
          </button>
        `);
      }
      updateLoadPrevVisibility(); // sync visibility with current scroll/top
    }
    function setLoadPrevLoading(loading) {
      const $btn = $('#load-prev-msgs');
      if (!$btn.length) return;
      $btn.prop('disabled', loading);
      $btn.find('.spinner-border').toggleClass('d-none', !loading);
    }

    // Dynamic near-top detection (2.5% of container height, clamped 8–64px)
    function computeTopThresholdPx() {
      const c = document.getElementById('chat-content');
      if (!c) return 16;
      const px = c.clientHeight * 0.025;
      return Math.max(8, Math.min(64, Math.round(px)));
    }
    function isNearTop() {
      const c = document.getElementById('chat-content');
      if (!c) return false;
      return c.scrollTop <= computeTopThresholdPx();
    }
    function updateLoadPrevVisibility() {
      const $btn = $('#load-prev-msgs');
      if (!$btn.length) return;
      if (!msgsHasNext) { $btn.addClass('d-none'); return; }
      if (isNearTop()) $btn.removeClass('d-none'); else $btn.addClass('d-none');
    }

    function resolveRole(m) {
      // Prefer explicit role strings
      const raw = (m.sender ?? m.role ?? m.author ?? '').toString().toLowerCase();
      if (raw === 'user' || raw === 'assistant') return raw;

      // Fallbacks for other shapes
      if (typeof m.is_user === 'boolean') return m.is_user ? 'user' : 'assistant';
      if (m.sender === userName) return 'user';   // legacy: sender was username
      return 'assistant';
    }

    function getText(m) {
      return m.text ?? m.content ?? m.message ?? '';
    }

    // Single message renderer (API sends sender=username; compare with localStorage userName)
    function renderMessageEl(m) {
      const role   = resolveRole(m);
      const cls    = role === 'user' ? 'user' : 'assistant';
      const avatar = role === 'assistant' ? 'AI' : '';
      const html   = marked.parse(getText(m));
      return `
        <div class="message ${cls}">
          <div class="message-avatar ${cls}">${avatar}</div>
          <div class="message-content">${html}</div>
        </div>
      `;
    }

    // Compute next PATH from meta (supports custom meta and DRF default)
    function deriveNextPathFromMeta(meta, chatId) {
      if (meta && meta.has_next) {
        if (meta.next_page) {
          return `/conversations/${chatId}/messages/?page=${meta.next_page}`;
        }
        if (meta.next) {
          try {
            const u = new URL(meta.next);
            let path = u.pathname + u.search;
            if (path.startsWith('/api/')) path = path.slice(4);
            return path;
          } catch {
            return meta.next.replace(/^\/api\//, '/');
          }
        }
        return `/conversations/${chatId}/messages/?page=2`;
      }
      if (meta && meta.next) {
        try {
          const u = new URL(meta.next);
          let path = u.pathname + u.search;
          if (path.startsWith('/api/')) path = path.slice(4);
          return path;
        } catch {
          return meta.next.replace(/^\/api\//, '/');
        }
      }
      return null;
    }

    /**
     * Fetch messages for a conversation.
     * mode = 'reset'   → load latest page (?page=1), bottom-align, show initial loader
     * mode = 'prepend' → call msgsNextPath (older page) and PREPEND without scroll jump
     */
    function fetchMessages(chatId, mode = 'reset') {
      const $chat = $('#chat-content');

      if (mode === 'reset') {
        msgsIsLoading = false;
        msgsHasNext   = false;
        msgsNextPath  = null;

        ensureMessageShell();
        $('#messages-list').empty();
        $('#load-prev-wrap').empty();

        // show initial loader while first page is loading
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

        // normalize meta (custom or DRF default)
        const meta = resp && (resp.meta ? resp.meta : (
          (typeof resp === 'object' && ('next' in resp || 'previous' in resp || 'count' in resp))
            ? { next: resp.next, previous: resp.previous, count: resp.count }
            : null
        ));

        const list = resp && resp.results
          ? resp.results
          : Array.isArray(resp) ? resp : [];

        const $list = $('#messages-list');

        if (mode === 'reset') {
          hideInitialMessagesLoader();
        }

        if (mode === 'reset' && list.length === 0) {
          $chat.html('<div class="text-center text-muted">Start a new conversation!</div>');
          return;
        }

        // API sends newest→oldest; reverse per page for chat UX
        const ascending = list.slice().reverse();

        if (mode === 'reset') {
          $list.empty(); // remove loader node before appending
          ascending.forEach(m => $list.append(renderMessageEl(m)));
          const c = document.getElementById('chat-content');
          c.scrollTop = c.scrollHeight; // bottom-align first load
        } else {
          // Keep viewport anchored when prepending
          const c = document.getElementById('chat-content');
          const before = c.scrollHeight;
          const prevScrollTop = c.scrollTop;
          $list.prepend(ascending.map(renderMessageEl).join(''));
          const after = c.scrollHeight;
          c.scrollTop = prevScrollTop + (after - before);
        }

        // Update next (older) page path
        const nextPath = deriveNextPathFromMeta(meta, chatId);
        msgsHasNext   = !!nextPath;
        msgsNextPath  = nextPath;

        renderLoadPrevButton();
        updateLoadPrevVisibility(); // recalc with dynamic threshold

      }, () => {
        msgsIsLoading = false;
        if (mode === 'prepend') setLoadPrevLoading(false);
        if (mode === 'reset') {
          hideInitialMessagesLoader();
          $chat.html('<div class="text-center text-danger">Failed to load messages.</div>');
        }
      });
    }

    function startNewChat() {
      currentChatId = null;
      $('.chat-item').removeClass('active');
      $('#chat-content')
        .empty()
        .append('<div class="text-center text-muted"><b>Start a new conversation!</b></div>');
    }

    function selectChat(chatId) {
      currentChatId = chatId;
      $('.chat-item').removeClass('active');
      $(`.chat-item[data-chat-id="${chatId}"]`).addClass('active');
      fetchMessages(chatId, 'reset');

      // Clear previous chat ID from span
      $('#chat-id-holder')
      .attr('data-current-conversation-id', '')
      .data('current-conversation-id', '');
  // Set new chat ID into span
      $('#chat-id-holder')
      .attr('data-current-conversation-id', chatId)
      .data('current-conversation-id', chatId);
    }

    // Holds raw text accumulating during token streaming
    let streamingText = '';
    let $streamBubble = null;
    let _rafPending = false;   // requestAnimationFrame gate

    function _ensureStreamBubble() {
      if ($streamBubble && $streamBubble.length) return;
      if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
      ensureMessageShell();
      const $list = $('#messages-list').length ? $('#messages-list') : $('#chat-content');
      $streamBubble = $(`
        <div class="message assistant">
          <div class="message-avatar assistant">AI</div>
          <div class="message-content" id="stream-content"></div>
        </div>`);
      $list.append($streamBubble);
    }

    // Throttled render — at most once per animation frame (~60 fps)
    function _scheduleStreamRender() {
      if (_rafPending) return;
      _rafPending = true;
      requestAnimationFrame(function () {
        _rafPending = false;
        const $c = $('#stream-content');
        if (!$c.length) return;
        $c.html(
          marked.parse(streamingText) +
          '<span class="stream-cursor"></span>'
        );
        scrollToBottom();
      });
    }

    function _sseFinalize(answer, uuid) {
      _rafPending = false; // cancel any pending frame

      if ($streamBubble && $streamBubble.length) {
        // Final clean render — remove cursor, fix any partial markdown
        $('#stream-content').html(marked.parse(answer || streamingText));
        $streamBubble = null;
      } else {
        if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
        ensureMessageShell();
        const $list = $('#messages-list').length ? $('#messages-list') : $('#chat-content');
        $list.append(renderMessageEl({ sender: 'assistant', text: answer }));
      }

      streamingText = '';
      isSubmitting = false;
      updateSendButton();
      $('#message-input').prop('disabled', false).focus();
      // Defer scroll until after the browser paints the full markdown content
      // (large response cards like Sales Summary can shift layout mid-render).
      requestAnimationFrame(() => scrollToBottom());

      if (!currentChatId && uuid) {
        currentChatId = uuid;
        fetchChats(false, () => {
          // Activate the new chat in the sidebar without reloading messages —
          // they are already correctly rendered in the DOM from the live stream.
          $('.chat-item').removeClass('active');
          $(`.chat-item[data-chat-id="${currentChatId}"]`).addClass('active');
          $('#chat-id-holder')
            .attr('data-current-conversation-id', currentChatId)
            .data('current-conversation-id', currentChatId);
        });
      }
    }

    async function sendMessage() {
      const $input = $('#message-input');
      const text = $input.val().trim();
      if (!text || isSubmitting) return;

      const token = getAuthToken();
      if (!token) { window.location.href = '/welcome'; return; }

      isSubmitting = true;
      streamingText = '';
      $streamBubble = null;
      $input.val('').css('height', 'auto').prop('disabled', true);
      updateSendButton();

      // User bubble
      ensureMessageShell();
      const $list = $('#messages-list').length ? $('#messages-list') : $('#chat-content');
      $list.append(renderMessageEl({ sender: userName, text }));
      scrollToBottom();

      // Typing indicator with live status line
      typingIndicator = $(`
        <div class="message assistant" id="sse-typing">
          <div class="message-avatar assistant">AI</div>
          <div class="message-content">
            <div class="typing-indicator">
              <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
            <div id="sse-status" style="font-size:.8rem;color:#888;margin-top:4px;min-height:1em;"></div>
          </div>
        </div>`);
      $list.append(typingIndicator);
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

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let eventName = 'message';
        let dataStr = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          const lines = buf.split('\n');
          buf = lines.pop(); // keep incomplete trailing line

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
                $('#sse-status').text(payload.message || '');
                scrollToBottom();

              } else if (eventName === 'token') {
                const chunk = payload.chunk || '';
                if (chunk) {
                  streamingText += chunk;
                  _ensureStreamBubble();
                  _scheduleStreamRender(); // throttled — max 60fps, cursor included
                }

              } else if (eventName === 'final') {
                const answer = (payload.answer || streamingText || '').trim() ||
                  'I prepared the results for you — see details below.';
                _sseFinalize(answer, payload.uuid);

              } else if (eventName === 'error') {
                const msg = (payload.message || 'Something went wrong.').trim();
                _sseFinalize(msg, payload.uuid);
              }

              eventName = 'message';
              dataStr = '';
            }
          }
        }

        // Stream ended without a final event — recover
        if (isSubmitting) {
          _sseFinalize(streamingText || 'I prepared the results — please check the conversation.', null);
        }

      } catch (err) {
        if ($streamBubble) { $streamBubble = null; }
        if (typingIndicator) { typingIndicator.remove(); typingIndicator = null; }
        streamingText = '';
        isSubmitting = false;
        updateSendButton();
        $input.prop('disabled', false).focus();
        Swal.fire({ icon: 'error', title: 'Error', text: 'Connection failed. Please try again.' });
      }
    }

    // Event bindings
    $('#send-btn').on('click', () => sendMessage());

    $('#message-input').on('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Enable/disable send button based on input content
    $('#message-input').on('input', function () {
      this.style.height = 'auto';
      this.style.height = this.scrollHeight + 'px';
      updateSendButton();
    });

    $('.chats-section').on('click', '.chat-item', function () {
      selectChat($(this).data('chat-id'));
    });

    $('.sidebar-menu .menu-item:contains("New chat")').on('click', startNewChat);

    // Click handler for the Load more chats button
    $('.chats-section').on('click', '#load-more-convos', function () {
      fetchChats(true);
    });

    // Load older messages when clicking the button
    $('#chat-content').on('click', '#load-prev-msgs', function () {
      if (currentChatId && msgsHasNext && !msgsIsLoading) {
        fetchMessages(currentChatId, 'prepend');
      }
    });

    // Show/hide Load previous based on scroll position (dynamic threshold)
    let topScrollTimer = null;
    $('#chat-content').on('scroll', function () {
      if (topScrollTimer) clearTimeout(topScrollTimer);
      topScrollTimer = setTimeout(updateLoadPrevVisibility, 50);
    });

    // Recalculate on resize as well
    let resizeTimer = null;
    $(window).on('resize', function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(updateLoadPrevVisibility, 100);
    });

    // Initialize
    const $sec = $('.chats-section');
    if ($sec.children('.chats-header').length === 0) {
      $sec.prepend('<div class="chats-header">Chats</div>');
    }

    fetchChats(false); // first page
    startNewChat();
    updateSendButton();

    // ===== Search chats =====
    $('#search-chats-btn').on('click', function () {
      const $wrap = $('#chat-search-wrap');
      $wrap.toggle();
      if ($wrap.is(':visible')) {
        $('#chat-search-input').val('').trigger('input').focus();
      } else {
        // restore all items when closed
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
  });

  // Dropdown & Misc Handlers
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
    const $item = $(`.chat-item[data-chat-id="${chatId}"]`);
    const $title = $item.find('.chat-title');
    const current = $title.text();

    // Replace title span with an input
    const $input = $(`<input type="text" class="chat-rename-input"
      value="${current.replace(/"/g, '&quot;')}"
      style="width:100%;border:1px solid #19c37d;border-radius:4px;padding:2px 6px;
             font-size:13px;outline:none;background:#fff;color:#374151;" />`);
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
      if (e.key === 'Enter') { e.preventDefault(); saveRename(); }
      if (e.key === 'Escape') { $input.replaceWith(`<div class="chat-title">${current}</div>`); }
    });
    $input.on('blur', saveRename);
  }

  function archiveChat(chatId) {
    if (confirm('Archive this chat?')) {
      $(`.chat-item[data-chat-id="${chatId}"]`).fadeOut(300, function () {
        $(this).remove();
      });
    }
    $('.chat-dropdown').removeClass('show');
  }

  function deleteChat(chatId) {
    if (confirm('Delete this chat?')) {
      $(`.chat-item[data-chat-id="${chatId}"]`).fadeOut(300, function () {
        $(this).remove();
      });
    }
    $('.chat-dropdown').removeClass('show');
  }

  function toggleUserDropdown() {
    $('.chat-dropdown').removeClass('show');
    $('#user-dropdown').toggleClass('show');
  }

  function logOut() {
    const token = localStorage.getItem('auth_token');
    const refreshToken = localStorage.getItem('refresh_token');

    $.ajax({
      url: apiBase + '/user/logout/',
      method: 'POST',
      headers: { Authorization: 'Bearer ' + token },
      data: JSON.stringify({ refresh_token: refreshToken }),
      contentType: 'application/json',
      success: function () {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
        localStorage.clear();
        window.location.href = '/welcome';
      },
      error: function (xhr) {
        let detail;
        try {
          detail = JSON.parse(xhr.responseText).detail;
        } catch {
          detail = xhr.responseText;
        }
        if (detail && detail.includes('Token is blacklisted')) {
          localStorage.removeItem('auth_token');
          localStorage.removeItem('refresh_token');
          localStorage.clear();
          window.location.href = '/welcome';
        } else {
          Swal.fire({
            icon: 'error',
            title: 'Oops...',
            text: 'Logout failed. Please try again.',
          });
        }
      },
    });
  }

  function openSettings() {
    $('.chat-dropdown, .user-dropdown').removeClass('show');
    const modalEl = document.getElementById('settingsModal');
    const bsModal = new bootstrap.Modal(modalEl);
    bsModal.show();
  }

  function saveSettings() {
    // Add your settings save logic here
  }

  // Close dropdowns when clicking outside
  $(document).on('click', function (e) {
    if (!$(e.target).closest('.chat-item, .user-info').length) {
      $('.chat-dropdown, .user-dropdown').removeClass('show');
    }
  });