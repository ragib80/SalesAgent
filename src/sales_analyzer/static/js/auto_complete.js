(function () {
  let setConversationId = null;

  let justInsertedOnce = false;
  let suppressGenericOnce = false;

  const INPUT_SELECTOR = '#message-input';
  const SEND_BTN_SELECTOR = '#send-btn';
  const AUTOCOMPLETE_URL = '/api/sales/autocomplete/';
  const AI_SUGGEST_URL = '/api/sales/prompt-suggestions/';
  const AI_SUGGEST_MAX_LEN = 20; //  NEW: cap length to avoid calling when > 20 chars
  const AI_SUGGEST_MIN_LEN = 4; //  
  const JWT_TOKEN = localStorage.getItem("auth_token");
  if (JWT_TOKEN) $.ajaxSetup({ headers: { 'Authorization': 'Bearer ' + JWT_TOKEN } });

  // Grab the raw token at the caret as last-resort
  function caretToken() {
    const $ta = $('#message-input');
    const full = $ta.val();
    const pos = $ta[0].selectionStart;
    const before = full.slice(0, pos);
    const m = before.match(/(?:^|[\s,])([^\s,]+)$/);
    return m ? m[1] : '';
  }

  /* ---------------- Labels & regex ---------------- */
  const FIELD_LABELS = [
    "Dealer", "Brand", "Product Name", "Material Group", "Division", "Company Code", "Sales Org",
    "Distribution Channel", "Business Area", "Credit Control Area", "Dealer Group", "Account Group",
    "Sales Group", "Sales Office", "Payer ID", "Product Code", "Volume Unit", "Business Group",
    "Territory", "Sales Zone", "Date", "Dealer Code", "Invoice Number"
  ];
  const fieldAlt = FIELD_LABELS
    .slice()
    .sort((a, b) => b.length - a.length)
    .map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|');

  const findAnyFieldContext = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*([^,:]*)$`, 'i');
  const findNewFieldStart = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*$`, 'i');

  /* ---------------- Elements ---------------- */
  const $ta = $(INPUT_SELECTOR);
  const $wrapper = $('.input-wrapper');
  const $inputArea = $('.input-area');

  // Create a BODY-LEVEL portal for the ghost card (prevents clipping/stacking issues)
  let $ghostPortal = $('#ghost-portal');
  if (!$ghostPortal.length) {
    $ghostPortal = $('<div id="ghost-portal" />').appendTo('body');
  }
  $ghostPortal.css({
    position: 'fixed',
    zIndex: 10020,
    display: 'none',
    pointerEvents: 'auto'
  });

  // Smooth lift for the input bar when smart dropdown needs space
  $inputArea.css({ transition: 'transform 160ms ease, opacity 160ms ease', willChange: 'transform' });

  /* ---------------- Helpers ---------------- */
  function normalizeFieldName(field) {
    const map = {
      "dealer": "Dealer", "brand": "Brand", "product name": "Product Name", "product": "Product",
      "material group": "Material Group", "division": "Division", "company code": "Company Code",
      "sales org": "Sales Org", "distribution channel": "Distribution Channel",
      "business area": "Business Area", "credit control area": "Credit Control Area",
      "dealer group": "Dealer Group", "account group": "Account Group",
      "sales group": "Sales Group", "sales office": "Sales Office", "payer id": "Payer ID",
      "product code": "Product Code", "volume unit": "Volume Unit", "business group": "Business Group",
      "territory": "Territory", "sales zone": "Sales Zone", "zone": "Sales Zone", "date": "Date",
      "dealer code": "Dealer Code", "invoice number": "Invoice Number"
    };
    return map[field?.toLowerCase()] || field;
  }

  function highlight(t, term) {
    if (!term) return t;
    try {
      return t.replace(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig'), '<span class="hl">$1</span>');
    } catch {
      return t;
    }
  }

  function setSendDisabled(v) { $(SEND_BTN_SELECTOR).prop('disabled', !!v); }
  function getWrapRect() { return $wrapper[0]?.getBoundingClientRect?.() || { top: 0, left: 0, width: 0, bottom: 0 }; }
  function hideGhost() { $ghostPortal.stop(true, true).fadeOut(120); }
  function setInputLift(px) { $inputArea.css('transform', `translateY(${-Math.max(0, px)}px)`); }

  /* ---------- Generic AJAX helper for autocomplete ---------- */
  function ajaxFetch(field, q, page, cb) {
    $.getJSON(AUTOCOMPLETE_URL, { field, q, page })
      .done(resp => {
        const items = (resp.results || []).map(r => ({ id: r.id, text: r.text, meta: r.meta || null }));
        cb({ items, more: !!(resp.pagination && resp.pagination.more) });
      })
      .fail(() => cb({ items: [], more: false }));
  }

  /* ---------- Single attachInfiniteScroll (namespaced optional) ---------- */
  function attachInfiniteScroll(dropdownEl, loader, ns = 'inf') {
    const $dd = $(dropdownEl);
    $dd.off('scroll.' + ns).on('scroll.' + ns, function () {
      if (this.scrollTop + this.clientHeight + 8 >= this.scrollHeight) loader();
    });
  }

  function hl(text, term) {
    if (!term) return text;
    try {
      const rx = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      return text.replace(rx, '<span class="hl">$1</span>');
    } catch { return text; }
  }

  function cardHTML(main, meta) {
    return `
      <div class="card suggest-card">
        <div class="card-body py-2 px-3">
          <div class="d-flex justify-content-between align-items-center gap-2">
            <div class="suggest-title">${main}</div>
            ${meta ? `<small class="suggest-meta">${meta}</small>` : ``}
          </div>
        </div>
      </div>`;
  }

  /* --------- detect explicit label context near caret --------- */
  function explicitContext() {
    const text = $ta.val().slice(0, $ta[0].selectionStart);
    const ctx = text.match(findAnyFieldContext) || text.match(findNewFieldStart);
    if (ctx) {
      return {
        label: normalizeFieldName((ctx[1] || '').trim()),
        term: (ctx[2] || '').trim()
      };
    }
    return null;
  }

  /* ---------------- Smart suggestions (textcomplete) ---------------- */
  let curField = null, curTerm = '', curPage = 1, hasMore = false, currentRequest = null;

  function searchField(field, q, page, cb) {
    $.getJSON(AUTOCOMPLETE_URL, { field: normalizeFieldName(field), q, page })
      .done(resp => {
        const items = (resp.results || []).map(r => ({ id: r.id, text: r.text, meta: r.meta || null }));
        cb({ items, more: !!(resp.pagination && resp.pagination.more) });
      })
      .fail(() => cb({ items: [], more: false }));
  }

  // ---------- 0) Explicit "Label term" strategy (e.g., "Dealer del") ----------
  (function labeledFieldStrategy () {
    // ", Dealer del" | "Dealer: del" | "Dealer del"
    const labeledRx = new RegExp(`(?:^|[\\s,])\\s*(${fieldAlt})\\s*:?\\s*([^,\\s]{1,})$`, 'i');
    let term = '', page = 1, more = false, label = '';

    $ta.textcomplete([{
      match: labeledRx,
      index: 2,
      cache: false,
      search: function (t, callback) {
        const m = ($ta.val().slice(0, $ta[0].selectionStart) || '').match(labeledRx);
        if (!m) { callback([]); return; }

        label = normalizeFieldName((m[1] || '').trim());
        term  = (m[2] || '').trim();
        if (!label || !term) { callback([]); return; }

        page = 1;
        ajaxFetch(label, term, page, ({ items, more: mMore }) => {
          more = mMore; callback(items);

          requestAnimationFrame(() => {
            const dd = document.querySelector('.textcomplete-dropdown');
            if (!dd) return;
            attachInfiniteScroll(dd, () => {
              if (!more) return;
              ajaxFetch(label, term, ++page, ({ items: next, more: m2 }) => {
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x => {
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>
                    ${cardHTML(hl(x.text, term), label)}
                  </a>`;
                  ul.appendChild(li);
                });
              });
            }, 'infLabeled');
          });
        });
      },
      replace: function (item) {
        const result = specialReplace(label, item.text);
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: (item) => `
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">${label}</small>
            </div>
          </div>
        </div>`
    }], { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });
  })();

  $ta.textcomplete([{
    match: /([^\s,].*)$/,
    index: 1,
    cache: false,
    context: function () { return false; }, // keep disabled by default
    search: function (term, callback) {
      if (currentRequest) { try { currentRequest.abort(); } catch { } }
      const text = $ta.val().slice(0, $ta[0].selectionStart);

      const ctx = text.match(findAnyFieldContext);
      if (ctx) {
        curField = ctx[1].trim(); curTerm = (ctx[2] || '').trim(); curPage = 1;
        if (curTerm.length >= 1) {
          currentRequest = searchField(curField, curTerm, curPage, ({ items, more }) => {
            hasMore = more; callback(items);
            requestAnimationFrame(() => {
              const dd = document.querySelector('.textcomplete-dropdown');
              if (!dd) return;
              attachInfiniteScroll(dd, () => {
                if (!hasMore) return;
                searchField(curField, curTerm, ++curPage, ({ items: next, more: m2 }) => {
                  hasMore = m2; const ul = dd.querySelector('ul'); if (!ul) return;
                  next.forEach(x => {
                    const li = document.createElement('li'); li.className = 'textcomplete-item';
                    li.innerHTML = `
                      <a><div class="card suggest-card"><div class="card-body py-2 px-3">
                        <div class="d-flex justify-content-between align-items-center gap-2">
                          <div class="suggest-title">${highlight(x.text, curTerm)}</div>
                          ${x.meta ? `<small class="suggest-meta text-truncate">${x.meta}</small>` : ''}
                        </div>
                      </div></div></a>`;
                    ul.appendChild(li);
                  });
                  layoutOverlaysNextPaint();
                });
              }, 'infGeneric');
            });
            layoutOverlaysNextPaint();
          });
          return;
        }
      }

      const nfs = text.match(findNewFieldStart);
      if (nfs) { curField = nfs[1].trim(); curTerm = ''; return callback([]); }

      const lastComma = text.lastIndexOf(',');
      if (lastComma !== -1) {
        const before = text.slice(0, lastComma);
        const fm = before.match(new RegExp(`\\b(${fieldAlt})\\s*:`, 'i'));
        if (fm) {
          curField = fm[1]; curTerm = text.slice(lastComma + 1).trim(); curPage = 1;
          if (curTerm.length >= 1) {
            currentRequest = searchField(curField, curTerm, curPage, ({ items, more }) => {
              hasMore = more; callback(items); layoutOverlaysNextPaint();
            });
            return;
          }
        }
      }

      curField = null; curTerm = ''; callback([]);
    },
    replace: function (item) {
      if (suppressGenericOnce) { suppressGenericOnce = false; return $ta.val(); }

      const full = $ta.val(), pos = $ta[0].selectionStart;
      const before = full.slice(0, pos), after = full.slice(pos);
      const label = curField ? curField.trim() : '';
      const labelRx = new RegExp(`\\b${label}\\b\\s*:?`, 'i');

      const ctx = before.match(findAnyFieldContext);
      if (ctx) {
        const fieldValue = (ctx[2] || '').trim();
        const start = before.lastIndexOf(fieldValue);
        return `${full.slice(0, start)}${item.text}${full.slice(pos)}`;
      }
      const lastComma = before.lastIndexOf(',');
      if (lastComma !== -1) {
        const b = full.slice(0, lastComma + 1), a = full.slice(pos);
        return (label && !b.match(labelRx)) ? `${b} ${label} ${item.text}, ${a}` : `${b} ${item.text}, ${a}`;
      }
      const nfs = before.match(findNewFieldStart);
      if (nfs) {
        const fname = nfs[1].trim(), pref = before.endsWith(',') ? '' : ', ';
        return `${full.slice(0, pos)}${pref}${fname} ${item.text}, ${after}`;
      }
      const pref = before.trim().endsWith(',') ? ' ' : ', ';
      return label ? `${before}${pref}${label} ${item.text}, ${after}` : `${before}${pref}${item.text}, ${after}`;
    },
    template: function (item) {
      const label = item.meta || (curField ? curField : '');
      return `<div class="card suggest-card">
            <div class="card-body py-2 px-3">
              <div class="d-flex align-items-center">
                <div class="suggest-title">${highlight(item.text, curTerm)}</div>
                ${label ? `<small class="suggest-meta ms-auto">${label}</small>` : ``}
              </div>
            </div>
          </div>`;
    }
  }],
    { maxCount: 15, debounce: 300, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
  );

  /* ---------------- Ghost suggestions (ABOVE input, via body portal) ---------------- */
  let aiTimer = null, lastQuery = '';

  function fallbackGhost(text) {
    const key = (text || '').trim().split(/\s+/).slice(-2).join(' ') || 'this';
    return [
      `Show me the sales trend for ${key} over the past year.`,
      `Compare ${key}'s sales with others in the same region.`,
      `Break down ${key}'s sales by product category.`
    ];
  }

  // SCROLLABLE, full-width suggestions; long text shown on hover via title=""
  function renderGhostHTML(sugs) {
    return `
      <div class="card ai-suggest-card shadow-sm"
           style="max-height: 200px; overflow-y: auto; overflow-x: hidden;">
        <div class="card-body py-2 px-3">
          <div class="fw-semibold mb-2 text-secondary small">Suggestions</div>
          ${sugs
            .map(
              s => `<div class="ai-suggest-item py-1 px-2 rounded-2 mb-1 text-truncate"
                        title="${s}" style="cursor:pointer;">
                      ${s}
                    </div>`
            )
            .join('')}
        </div>
      </div>`;
  }

  // Keep ghost full-width above input
  function placeGhostAboveInput() {
    if (!$ghostPortal.is(':visible')) return;

    // Temporarily make it measurable
    $ghostPortal.css({ visibility: 'hidden', display: 'block' });

    const r = getWrapRect();
    const left = r.left;
    const width = r.width;

    const viewportTop = 8;
    const anchorBottom = r.top - 10;

    // Enforce height cap here as well
    $ghostPortal.find('.ai-suggest-card').css({
      maxHeight: '200px',
      overflowY: 'auto'
    });

    const gh = $ghostPortal.outerHeight();
    const top = Math.max(viewportTop, anchorBottom - gh);

    $ghostPortal.css({ top, left, width, visibility: 'visible' });
  }

  async function fetchAISuggestions(text) {
    const chatId = $('#chat-id-holder').data('current-conversation-id');
    const q = (text || '').trim();
    if (!q) { hideGhost(); return; }
    lastQuery = q;

    // ----- LOADER -----
    $ghostPortal
      .html(`
        <div class="card ai-suggest-card shadow-sm" style="padding: 10px;">
          <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm text-secondary me-2" role="status"></div>
            <span class="text-secondary small">Generating Suggestions...</span>
          </div>
        </div>
      `)
      .stop(true, true)
      .fadeIn(120);

    placeGhostAboveInput();

    try {
      const resp = await $.ajax({
        url: AI_SUGGEST_URL, method: 'POST',
        data: JSON.stringify({ input_text: q, conversation_id: chatId }),
        contentType: 'application/json'
      });

      let suggestions = (resp && resp.suggestions) || [];
      if (!suggestions.length) suggestions = fallbackGhost(q);

      const still = ($ta.val() || '').trim();
      if (!still || still !== lastQuery) { hideGhost(); return; }

      $ghostPortal
        .html(renderGhostHTML(suggestions))
        .off('click', '.ai-suggest-item')
        .on('click', '.ai-suggest-item', function () {
          $ta.val($(this).text().trim()).trigger('input'); hideGhost();
        })
        .stop(true, true).fadeIn(120);

      placeGhostAboveInput();
    } catch (e) {
      console.log('AI suggest error:', e);
      $ghostPortal.find('.text-secondary').text('Could not load suggestions');
      placeGhostAboveInput();
    }
  }

  $ta.on('input', function () {
    const v = (this.value || '').trim();
    setSendDisabled(!v);
    clearTimeout(aiTimer);

    if (!v) { hideGhost(); return; }

    if (v.length >= AI_SUGGEST_MIN_LEN && v.length <= AI_SUGGEST_MAX_LEN) {
      aiTimer = setTimeout(() => fetchAISuggestions(v), 500);
    } else {
      hideGhost();
    }
  });

  /* ---------------- Layout: smart below (with lift), ghost above ---------------- */
  function placeDropdownBelowInput() {
    const $dd = $('.textcomplete-dropdown:visible');
    if (!$dd.length || !$wrapper.length) {
      setInputLift(0); return;
    }
    $dd.css({ position: 'fixed', visibility: 'hidden', display: 'block', height: '', maxHeight: '360px' });

    const r = getWrapRect();
    const ddH = $dd.outerHeight();
    const gap = 8;

    const spaceBelow = window.innerHeight - (r.bottom + gap);
    const liftNeeded = Math.max(0, ddH - spaceBelow + 8);
    const maxLift = Math.max(0, r.top - 16);
    const lift = Math.min(liftNeeded, maxLift);

    setInputLift(lift);
    const top = r.bottom - lift + gap;

    $dd.css({ top, left: r.left, width: r.width + 'px', visibility: 'visible', zIndex: 10010 });
  }

  function layoutOverlays() { placeDropdownBelowInput(); placeGhostAboveInput(); }
  function layoutOverlaysNextPaint() { requestAnimationFrame(() => requestAnimationFrame(layoutOverlays)); }

  $ta.on('textComplete:show textComplete:rendered textComplete:append textComplete:hide', layoutOverlaysNextPaint);
  $(window).on('resize scroll', layoutOverlaysNextPaint);
  $('.chat-content').on('scroll', layoutOverlaysNextPaint);

  $(document).on('click', (e) => {
    if (!$(e.target).closest('#ghost-portal, .input-wrapper, .textcomplete-dropdown').length) hideGhost();
  });
  $ta.on('blur', hideGhost);

  const observer = new MutationObserver(() => { if (!$('.textcomplete-dropdown:visible').length) setInputLift(0); });
  observer.observe(document.body, { childList: true, subtree: true });

  $(function () {
    hideGhost();
    setSendDisabled(true);
  });

  /* ---------- insertLabeled: Core deduplication logic ---------- */
  function insertLabeled(label, value) {
    const $ta = $('#message-input');
    let full = $ta.val();
    const pos = $ta[0].selectionStart;

    // Remove the raw token being typed (like "t005" or "deco")
    let before = full.slice(0, pos).replace(/(?:^|[\s,])([^\s,]+)$/, '').trim();
    const after = full.slice(pos).trim();

    const squish = s => (s || '').replace(/\s+/g, ' ').trim();

    const canonLabel = s => {
      s = squish(s);
      if (/^zone$/i.test(s)) return 'Sales Zone';
      if (/^sales\s*zone$/i.test(s)) return 'Sales Zone';
      if (/^division\s*code$/i.test(s)) return 'Division Code';
      if (/^division$/i.test(s)) return 'Division';
      if (/^territory$/i.test(s)) return 'Territory';
      if (/^material\s*group$/i.test(s)) return 'Material Group';
      return s.replace(/\b\w/g, c => c.toUpperCase());
    };

    const L = canonLabel(label);
    const V = squish(value);

    // Fields that support multiple values under same label
    const MULTI_VALUE_LABELS = new Set([
      "Division",
      "Division Code",
      "Material Group",
      "Territory",
      "Sales Zone"
    ]);

    const parts = before ? before.split(',').map(s => squish(s)).filter(Boolean) : [];

    let groups = {};  // { Label: [values] }
    let order = [];   // preserve label order

    // Build current groups
    for (let p of parts) {
      const match = p.match(/^([A-Za-z\s]+?)\s+(.+)$/);
      if (match) {
        let lbl = canonLabel(match[1]);
        let val = squish(match[2]);
        if (!groups[lbl]) {
          groups[lbl] = [];
          order.push(lbl);
        }
        if (!groups[lbl].includes(val)) {
          groups[lbl].push(val);
        }
      }
    }

    // Insert new value
    if (!groups[L]) {
      groups[L] = [V];
      order.push(L);
    } else {
      if (MULTI_VALUE_LABELS.has(L)) {
        if (!groups[L].includes(V)) groups[L].push(V);
      } else {
        groups[L] = [V];
      }
    }

    // Build output string
    let result = '';
    for (let lbl of order) {
      if (groups[lbl] && groups[lbl].length) {
        result += lbl + ' ' + groups[lbl].join(', ') + ', ';
      }
    }

    return result + (after ? after + ' ' : '');
  }

  function specialReplace(label, item) {
    const val = (item && (item.text || item.value || item.id)) ||
      (typeof item === 'string' ? item : '') ||
      caretToken();

    const result = insertLabeled(label.trim(), String(val).trim());
    return result;
  }

  /* ---------- 1) Territory: T\d+ ---------- */
  (function territoryStrategy() {
    let term = '', page = 1, more = false;
    $ta.textcomplete([{
      match: /(?:^|[\s,])([Tt]\d{1,})$/,
      index: 1,
      cache: false,
      search: function (t, callback) {
        // Guard: if user is inside a DIFFERENT explicit label, don't trigger quick territory search
        const ec = explicitContext();
        if (ec && ec.label && ec.label !== 'Territory') { callback([]); return; }

        term = t.trim(); page = 1;
        ajaxFetch('Territory', term, page, ({ items, more: m }) => {
          more = m; callback(items);
          requestAnimationFrame(() => {
            const dd = document.querySelector('.textcomplete-dropdown');
            if (!dd) return;
            attachInfiniteScroll(dd, () => {
              if (!more) return;
              ajaxFetch('Territory', term, ++page, ({ items: next, more: m2 }) => {
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x => {
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              });
            }, 'infTerritory');
          });
        });
      },
      replace: function (item) {
        const result = specialReplace('Territory', item.text);
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: item => `
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">Territory</small>
            </div>
          </div>
        </div>`
    }],
      { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
    );
  })();

  /* ---------- 2) Sales Zone: Z\d+ ---------- */
  (function szoneStrategy() {
    let term = '', page = 1, more = false;
    $ta.textcomplete([{
      match: /(?:^|[\s,])([Zz]\d{1,})$/,
      index: 1,
      cache: false,
      search: function (t, callback) {
        // Guard against explicit label that isn't Sales Zone
        const ec = explicitContext();
        if (ec && ec.label && ec.label !== 'Sales Zone') { callback([]); return; }

        term = t.trim(); page = 1;
        ajaxFetch('Sales Zone', term, page, ({ items, more: m }) => {
          more = m; callback(items);
          requestAnimationFrame(() => {
            const dd = document.querySelector('.textcomplete-dropdown');
            if (!dd) return;
            attachInfiniteScroll(dd, () => {
              if (!more) return;
              ajaxFetch('Sales Zone', term, ++page, ({ items: next, more: m2 }) => {
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x => {
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              });
            }, 'infSzone');
          });
        });
      },
      replace: function (item) {
        const result = specialReplace('Sales Zone', item.text); // explicit
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: item => `
      <div class="card suggest-card">
        <div class="card-body py-2 px-3">
          <div class="d-flex align-items-center">
            <div class="suggest-title">${hl(item.text, term)}</div>
            <small class="suggest-meta ms-auto">Sales Zone</small>
          </div>
        </div>
      </div>`
    }],
      { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
    );
  })();

  /* ---------- 3) Material Group ---------- */
  (function matklStrategy() {
    let term = '', page = 1, more = false;
    const rx = /(?:^|[\s,])((?:[Ff]\d+|[Bb][Pp]\d+|[Mm][Vv]\d+|[Bb]\d+|[Ss](?:[Oo])?\d+|\d{3,6}))$/;
    $ta.textcomplete([{
      match: rx,
      index: 1,
      cache: false,
      search: function (t, callback) {
        // Guard against explicit label that isn't Material Group
        const ec = explicitContext();
        if (ec && ec.label && ec.label !== 'Material Group') { callback([]); return; }

        term = t.trim(); page = 1;
        ajaxFetch('Material Group', term, page, ({ items, more: m }) => {
          more = m; callback(items);
          requestAnimationFrame(() => {
            const dd = document.querySelector('.textcomplete-dropdown');
            if (!dd) return;
            attachInfiniteScroll(dd, () => {
              if (!more) return;
              ajaxFetch('Material Group', term, ++page, ({ items: next, more: m2 }) => {
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x => {
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              });
            }, 'infMatkl');
          });
        });
      },
      replace: function (item) {
        const result = specialReplace('Material Group', item.text);
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: item => `
      <div class="card suggest-card">
        <div class="card-body py-2 px-3">
          <div class="d-flex align-items-center">
            <div class="suggest-title">${hl(item.text, term)}</div>
            <small class="suggest-meta ms-auto">Material Group</small>
          </div>
        </div>
      </div>`
    }],
      { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
    );
  })();

  /* ---------- 4) Division Names and Codes ---------- */
  (function divisionStrategy() {
    let term = '', page = 1, more = false;

    const DIVISION_NAMES = [
      "Decorative", "Industrial Paints", "Adhesive & Chemicals", "Powder Coating", "Wood Coating",
      "Marine Paints", "Vehicle Refinish", "Printing Ink", "Trading", "Texbond", "Service",
      "Construction Chemica", "Food Grade Can", "Industrial Grade Can", "Container Scraps",
      "Coil Coating", "Training Service"
    ];
    const DIVISION_CODES = new Set(["10", "20", "50", "40", "80", "30", "70", "11", "60", "55", "65", "90", "3", "2", "4", "95", "66"]);

    // Division Names (free-text)
    $ta.textcomplete([{
      match: /(?:^|[\s,])([A-Za-z][A-Za-z\s]{3,})$/,
      index: 1,
      cache: false,
      search: function (t, callback) {
        // If user is inside ANY explicit label (Dealer, Territory, etc.), do NOT treat as Division free-text
        const ec = explicitContext();
        if (ec && ec.label) { callback([]); return; }

        term = (t || '').trim();
        if (term.length < 4) { callback([]); return; }

        const lower = term.toLowerCase();
        const local = DIVISION_NAMES
          .filter(n => n.toLowerCase().startsWith(lower))
          .slice(0, 6)
          .map(n => ({ id: n, text: n, meta: 'Division' }));

        callback(local);

        page = 1;
        ajaxFetch('Division', term, page, ({ items, more: m }) => {
          more = m;
          const seen = new Set(local.map(i => i.text.toLowerCase()));
          const merged = local.concat(items.filter(i => !seen.has(i.text.toLowerCase())));
          callback(merged);

          requestAnimationFrame(() => {
            const dd = document.querySelector('.textcomplete-dropdown');
            if (!dd) return;
            attachInfiniteScroll(dd, () => {
              if (!more) return;
              ajaxFetch('Division', term, ++page, ({ items: next, more: m2 }) => {
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x => {
                  const key = x.text.toLowerCase();
                  if (seen.has(key)) return;
                  seen.add(key);
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), 'Division')}</a>`;
                  ul.appendChild(li);
                });
              });
            }, 'infDivName');
          });
        });
      },
      replace: function (item) {
        const result = specialReplace('Division', item.text);
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: item => `
      <div class="card suggest-card">
        <div class="card-body py-2 px-3">
          <div class="d-flex align-items-center">
            <div class="suggest-title">${hl(item.text, term)}</div>
            <small class="suggest-meta ms-auto">Division</small>
          </div>
        </div>
      </div>`
    }],
      { maxCount: 15, debounce: 140, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
    );

    // Division Codes (digits)
    $ta.textcomplete([{
      match: /(?:^|[\s,])(\d{1,2})$/,
      index: 1,
      cache: false,
      search: function (t, callback) {
        // If inside an explicit (different) label, ignore
        const ec = explicitContext();
        if (ec && ec.label && ec.label !== 'Division Code') { callback([]); return; }

        const code = (t || '').trim();
        if (DIVISION_CODES.has(code)) {
          callback([{ id: code, text: code, meta: 'Division Code' }]);
        } else {
          callback([]);
        }
      },
      replace: function (item) {
        const result = specialReplace('Division Code', item.text);
        const $input = $('#message-input');
        $input.val(result).trigger('input');
        $input.trigger('textComplete:hide');
      },
      template: item => cardHTML(item.text, 'Division Code')
    }],
      { maxCount: 8, debounce: 80, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' }
    );
  })();

})();
