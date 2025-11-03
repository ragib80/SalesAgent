(function () {
  /* =================== Config =================== */
  const INPUT_SELECTOR = '#message-input';
  const SEND_BTN_SELECTOR = '#send-btn';
  const AUTOCOMPLETE_URL = '/api/sales/autocomplete/';
  const AI_SUGGEST_URL = '/api/sales/prompt-suggestions/';
  const AI_SUGGEST_MAX_LEN = 20;
  const AI_SUGGEST_MIN_LEN = 4;

  const JWT_TOKEN = localStorage.getItem('auth_token');
  if (JWT_TOKEN) $.ajaxSetup({ headers: { Authorization: 'Bearer ' + JWT_TOKEN } });

  /* =================== Globals =================== */
  const $ta = $(INPUT_SELECTOR);
  const $wrapper = $('.input-wrapper');
  const $inputArea = $('.input-area');

  // Ghost portal (AI suggestions above input)
  let $ghostPortal = $('#ghost-portal');
  if (!$ghostPortal.length) $ghostPortal = $('<div id="ghost-portal" />').appendTo('body');

  const Z_DROPDOWN = 10050;  // autocomplete
  const Z_GHOST    = 10060;  // suggestions
  const MIN_GAP    = 12;

  $ghostPortal.css({ position: 'fixed', zIndex: Z_GHOST, display: 'none', pointerEvents: 'auto' });

  // Keep the input area smoothly liftable
  $inputArea.css({ transition: 'transform 160ms ease, opacity 160ms ease', willChange: 'transform' });

  // Stale guard + cancellable requests
  const __inflight = Object.create(null); // ns -> jqXHR
  let __epoch = 0;
  function __cancelNS(ns){ const h = __inflight[ns]; if (h && h.abort) { try{h.abort();}catch{} } __inflight[ns]=null; }

  /* =================== Helpers =================== */
  const FIELD_LABELS = [
    'Dealer','Brand','Product Name','Material Group','Division','Company Code','Sales Org',
    'Distribution Channel','Business Area','Credit Control Area','Dealer Group','Account Group',
    'Sales Group','Sales Office','Payer ID','Product Code','Volume Unit','Business Group',
    'Territory','Sales Zone','Date','Dealer Code','Invoice Number'
  ];
  const fieldAlt = FIELD_LABELS.slice().sort((a,b)=>b.length-a.length)
    .map((s)=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|');

  const findAnyFieldContext = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*([^,:]*)$`, 'i');
  const findNewFieldStart   = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*$`, 'i');

  function normalizeFieldName(field){
    const map = {
      dealer:'Dealer', brand:'Brand', 'product name':'Product Name', product:'Product',
      'material group':'Material Group', division:'Division', 'company code':'Company Code',
      'sales org':'Sales Org', 'distribution channel':'Distribution Channel', 'business area':'Business Area',
      'credit control area':'Credit Control Area', 'dealer group':'Dealer Group', 'account group':'Account Group',
      'sales group':'Sales Group', 'sales office':'Sales Office', 'payer id':'Payer ID',
      'product code':'Product Code', 'volume unit':'Volume Unit', 'business group':'Business Group',
      territory:'Territory', 'sales zone':'Sales Zone', zone:'Sales Zone', date:'Date',
      'dealer code':'Dealer Code', 'invoice number':'Invoice Number'
    };
    return map[field?.toLowerCase()] || field;
  }

  function caretToken(){
    const full = $ta.val(); const pos = $ta[0].selectionStart; const before = full.slice(0,pos);
    const m = before.match(/(?:^|[\s,])([^\s,]+)$/); return m ? m[1] : '';
  }

  function highlight(t, term){
    if (!term) return t;
    try { return t.replace(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`,'ig'), '<span class="hl">$1</span>'); }
    catch { return t; }
  }
  const hl = highlight;

  function setSendDisabled(v){ $(SEND_BTN_SELECTOR).prop('disabled', !!v); }
  function getWrapRect(){ return $wrapper[0]?.getBoundingClientRect?.() || { top:0,left:0,width:0,bottom:0 }; }

  function hideGhost(){ $ghostPortal.stop(true,true).fadeOut(120); }
  function setInputLift(px){ $inputArea.css('transform', `translateY(${-Math.max(0,px)}px)`); }
  function escapeRx(s){ return String(s).replace(/[.*+?^${}()|[\]\\]/g,'\\$&'); }

  // Clean trailing context from a label term, support quoted values
  function cleanFieldTerm(label, raw){
    let t = (raw || '').trim();
    if (!t) return '';
    if (t.startsWith('"')){
      const last = t.lastIndexOf('"');
      if (last > 0){ const inner = t.slice(1,last); if (inner.trim()) return inner.trim(); t = t.slice(1); }
      else t = t.slice(1);
    }
    t = t.replace(/[.;!?\n]+.*$/, '').trim();
    t = t.replace(/\b(?:in|for|from|to|between|of|on|by|within|over|during|until|through|throughout|vs|versus|than|after|before|with|without|and|or|the|a|at|as)\b.*$/i,'').trim();
    t = t.replace(/\b(?:in\s+)?(?:last|next|this)\s+\d{0,4}\s*(?:day|days|week|weeks|month|months|year|years|quarter|quarters|q[1-4])\b.*$/i,'').trim();
    return t;
  }

  function explicitContext(){
    const text = $ta.val().slice(0, $ta[0].selectionStart);
    const ctx  = text.match(findAnyFieldContext) || text.match(findNewFieldStart);
    if (!ctx) return null;
    return { label: normalizeFieldName((ctx[1]||'').trim()), term: (ctx[2]||'').trim() };
  }

  /* =================== Networking =================== */
  function ajaxFetch(field, q, page, cb, ns='default', epochGuard=null){
    __cancelNS(ns);
    const xhr = $.getJSON(AUTOCOMPLETE_URL, { field, q, page })
      .done((resp)=>{
        if (epochGuard !== null && epochGuard !== __epoch) return;
        const items = (resp.results || []).map((r)=>({
          id:r.id, text:r.text, meta:r.meta || null, commit:r.commit || r.insert || r.value || null
        }));
        cb({ items, more: !!(resp.pagination && resp.pagination.more) });
      })
      .fail(()=>{ if (epochGuard===null || epochGuard===__epoch) cb({ items:[], more:false }); })
      .always(()=>{ if (__inflight[ns] === xhr) __inflight[ns] = null; });
    __inflight[ns] = xhr;
    return xhr;
  }

  function searchField(field, q, page, cb, ns='generic', epochGuard=null){
    return ajaxFetch(normalizeFieldName(field), q, page, cb, ns, epochGuard);
  }

  function attachInfiniteScroll(dropdownEl, loader, ns='inf'){
    const $dd = $(dropdownEl);
    $dd.off('scroll.'+ns).on('scroll.'+ns, function(){
      if (this.scrollTop + this.clientHeight + 8 >= this.scrollHeight) loader();
    });
  }

  function cardHTML(main, meta){
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

  /* =================== Insertion =================== */
  function insertLabeled(label, value){
    const $i = $ta; let full = $i.val(); const pos = $i[0].selectionStart;
    let before = full.slice(0,pos).replace(/(?:^|[\s,])([^\s,]+)$/,'').trim();
    const after = full.slice(pos).trim();

    const squish = (s)=>(s||'').replace(/\s+/g,' ').trim();
    const canonLabel = (s)=>{
      s = squish(s);
      if (/^zone$/i.test(s)) return 'Sales Zone';
      if (/^sales\s*zone$/i.test(s)) return 'Sales Zone';
      if (/^division\s*code$/i.test(s)) return 'Division Code';
      if (/^division$/i.test(s)) return 'Division';
      if (/^territory$/i.test(s)) return 'Territory';
      if (/^material\s*group$/i.test(s)) return 'Material Group';
      return s.replace(/\b\w/g, (c)=>c.toUpperCase());
    };
    const L = canonLabel(label);
    const V = squish(value);

    const MULTI = new Set(['Division','Division Code','Material Group','Territory','Sales Zone']);
    const parts = before ? before.split(',').map(s=>squish(s)).filter(Boolean) : [];

    const groups = {}; const order = [];
    for (const p of parts){
      const m = p.match(new RegExp(`^(${fieldAlt})\\s+(.+)$`,'i'));
      if (!m) continue;
      const lbl = canonLabel(m[1]); const val = squish(m[2]);
      if (!groups[lbl]){ groups[lbl]=[]; order.push(lbl); }
      if (!groups[lbl].includes(val)) groups[lbl].push(val);
    }

    if (!groups[L]){ groups[L]=[V]; order.push(L); }
    else { if (MULTI.has(L)){ if (!groups[L].includes(V)) groups[L].push(V); } else { groups[L]=[V]; } }

    let out = '';
    for (const lbl of order){ if (groups[lbl]?.length) out += `${lbl} ${groups[lbl].join(', ')}, `; }
    return out + (after ? after + ' ' : '');
  }

  function specialReplace(label, item){
    const L = normalizeFieldName(String(label||'').trim());

    const pickRaw = (x)=>!x ? '' : (typeof x === 'string' ? x : (x.commit || x.text || x.value || x.id || ''));
    const fallback = caretToken();
    const raw = pickRaw(item) || fallback;
    let commit = raw;

    if (/^Dealer$/i.test(L)){
      commit = raw.replace(/\s*[-–—|]\s.*$/, '').replace(/,\s*$/, '').trim();
    }

    // Try to replace the *active* "Label term" segment (to prevent duplicates)
    const $i = $ta; let full = $i.val(); const pos = $i[0].selectionStart;
    const left  = full.slice(0,pos);
    const right = full.slice(pos);
    const rx = new RegExp(`(^|[\\s,])\\s*(${escapeRx(L)})\\s*:?\\s*([^,\\n]*)$`, 'i');
    const m  = rx.exec(left);

    if (m){
      const prefix = left.slice(0, m.index);
      const delim  = m[1] || ' ';
      const replacedLeft = `${prefix}${delim}${L} ${String(commit).trim()}, `;
      full = replacedLeft + right.replace(/^(\s*),\s*/, '$1');
      return full;
    }
    // Fallback: safe merge
    return insertLabeled(L, String(commit).trim());
  }

  /* =================== Autocomplete wiring (smart) =================== */
  let curField = null, curTerm = '', curPage = 1, hasMore = false, currentRequest = null;

  // Keep dropdown + ghost both working; just re-layout when dropdown changes
  $ta.off('textComplete:show textComplete:rendered textComplete:append textComplete:hide');
  $ta.on('textComplete:show textComplete:rendered textComplete:append textComplete:hide', () => {
    layoutOverlaysNextPaint();
  });

  // 0) Explicit "Label term" strategy (e.g., "Dealer Del...")
  (function labeledFieldStrategy(){
    const labeledRx = new RegExp(`(?:^|[\\s,])\\s*(${fieldAlt})\\s*:?\\s*([^,\\n]{1,})$`, 'i');
    let term='', page=1, more=false, label='';

    $ta.textcomplete([{
      match: labeledRx, index: 2, cache: false,
      search: function(t, callback){
        const m = ($ta.val().slice(0,$ta[0].selectionStart)||'').match(labeledRx);
        if (!m){ callback([]); return; }

        label = normalizeFieldName((m[1]||'').trim());
        term  = cleanFieldTerm(label, (m[2]||'').trim());
        if (!label || !term){ callback([]); return; }

        const epoch = ++__epoch; page = 1;
        ajaxFetch(label, term, page, ({items, more:mMore})=>{
          if (epoch !== __epoch) return;
          more = mMore; callback(items);

          requestAnimationFrame(()=>{
            const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
            attachInfiniteScroll(dd, ()=>{
              if (epoch !== __epoch || !more) return;
              ajaxFetch(label, term, ++page, ({items:next, more:m2})=>{
                if (epoch !== __epoch) return;
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x=>{
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), label)}</a>`;
                  ul.appendChild(li);
                });
              }, 'labeled', epoch);
            }, 'infLabeled');
          });
        }, 'labeled', epoch);
      },
      replace: function(item){
        const result = specialReplace(label, item);
        $ta.val(result).trigger('input');
        $ta.trigger('textComplete:hide');
      },
      template: (item)=>`
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

  // 1) Generic catch-all (context-aware)
  $ta.textcomplete([{
    match:/([^\s,].*)$/, index:1, cache:false,
    context: ()=> false,
    search: function(term, callback){
      if (currentRequest){ try{ currentRequest.abort(); }catch{} }
      const text = $ta.val().slice(0, $ta[0].selectionStart);

      const ctx = text.match(findAnyFieldContext);
      if (ctx){
        curField = ctx[1].trim();
        curTerm  = cleanFieldTerm(curField, (ctx[2]||'').trim());
        curPage  = 1;

        if (curTerm.length >= 1){
          const epoch = ++__epoch;
          currentRequest = searchField(curField, curTerm, curPage, ({items, more})=>{
            if (epoch !== __epoch) return;
            hasMore = more; callback(items);

            requestAnimationFrame(()=>{
              const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
              dd.style.zIndex = Z_DROPDOWN;
              attachInfiniteScroll(dd, ()=>{
                if (epoch !== __epoch || !hasMore) return;
                searchField(curField, curTerm, ++curPage, ({items:next, more:m2})=>{
                  if (epoch !== __epoch) return;
                  hasMore = m2;
                  const ul = dd.querySelector('ul'); if (!ul) return;
                  next.forEach(x=>{
                    const li = document.createElement('li');
                    li.className = 'textcomplete-item';
                    li.innerHTML = `
                      <a><div class="card suggest-card"><div class="card-body py-2 px-3">
                        <div class="d-flex justify-content-between align-items-center gap-2">
                          <div class="suggest-title">${hl(x.text, curTerm)}</div>
                          ${x.meta ? `<small class="suggest-meta text-truncate">${x.meta}</small>` : ''}
                        </div>
                      </div></div></a>`;
                    ul.appendChild(li);
                  });
                  layoutOverlaysNextPaint();
                }, 'generic', epoch);
              }, 'infGeneric');
            });
            layoutOverlaysNextPaint();
          }, 'generic', epoch);
          return;
        }
      }

      const nfs = text.match(findNewFieldStart);
      if (nfs){ curField = nfs[1].trim(); curTerm = ''; return callback([]); }

      const lastComma = text.lastIndexOf(',');
      if (lastComma !== -1){
        const before = text.slice(0, lastComma);
        const fm = before.match(new RegExp(`\\b(${fieldAlt})\\s*:`, 'i'));
        if (fm){
          curField = fm[1];
          curTerm  = cleanFieldTerm(curField, text.slice(lastComma+1).trim());
          curPage  = 1;
          if (curTerm.length >= 1){
            const epoch = ++__epoch;
            currentRequest = searchField(curField, curTerm, curPage, ({items, more})=>{
              if (epoch !== __epoch) return;
              hasMore = more; callback(items);
              layoutOverlaysNextPaint();
            }, 'generic', epoch);
            return;
          }
        }
      }

      curField = null; curTerm = ''; callback([]);
    },
    replace: function(item){
      const label = curField ? curField.trim() : '';
      const result = specialReplace(label, item);
      $ta.val(result).trigger('input');
      $ta.trigger('textComplete:hide');
    },
    template: function(item){
      const label = item.meta || (curField ? curField : '');
      return `<div class="card suggest-card">
        <div class="card-body py-2 px-3">
          <div class="d-flex align-items-center">
            <div class="suggest-title">${hl(item.text, curTerm)}</div>
            ${label ? `<small class="suggest-meta ms-auto">${label}</small>` : ``}
          </div>
        </div>
      </div>`;
    }
  }], { maxCount: 15, debounce: 300, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });

  /* =================== Ghost Suggestions (above input) =================== */
  let aiTimer = null, lastQuery = '';

  function fallbackGhost(text){
    const key = (text || '').trim().split(/\s+/).slice(-2).join(' ') || 'this';
    return [
      `Show me the sales trend for ${key} over the past year.`,
      `Compare ${key}'s sales with others in the same region.`,
      `Break down ${key}'s sales by product category.`
    ];
  }

  function renderGhostHTML(sugs){
    return `
      <div class="card ai-suggest-card shadow-sm" style="max-height:200px; overflow-y:auto; overflow-x:hidden; pointer-events:auto;">
        <div class="card-body py-2 px-3">
          <div class="fw-semibold mb-2 text-secondary small">Suggestions</div>
          ${sugs.map(s => `<div class="ai-suggest-item py-1 px-2 rounded-2 mb-1 text-truncate" title="${s}" style="cursor:pointer;">${s}</div>`).join('')}
        </div>
      </div>`;
  }

  function placeGhostAboveInput(){
    if (!$ghostPortal.is(':visible')) return;
    $ghostPortal.css({ visibility:'hidden', display:'block' });

    const r = getWrapRect();
    const width = r.width; const left = r.left;

    // height already capped to 200 via CSS
    const gh = $ghostPortal.outerHeight();
    const top = Math.max(8, r.top - MIN_GAP - gh);

    $ghostPortal.css({ top, left, width, visibility:'visible', zIndex: Z_GHOST });
  }

  async function fetchAISuggestions(text){
    const chatId = $('#chat-id-holder').data('current-conversation-id');
    const q = (text || '').trim();
    if (!q){ hideGhost(); return; }
    lastQuery = q;

    $ghostPortal.html(`
      <div class="card ai-suggest-card shadow-sm" style="padding:10px;">
        <div class="d-flex align-items-center">
          <div class="spinner-border spinner-border-sm text-secondary me-2" role="status"></div>
          <span class="text-secondary small">Generating Suggestions...</span>
        </div>
      </div>`).stop(true,true).fadeIn(120);

    placeGhostAboveInput();

    try{
      const resp = await $.ajax({
        url: AI_SUGGEST_URL, method: 'POST',
        data: JSON.stringify({ input_text: q, conversation_id: chatId }),
        contentType: 'application/json'
      });

      let suggestions = (resp && resp.suggestions) || [];
      if (!suggestions.length) suggestions = fallbackGhost(q);

      const still = ($ta.val() || '').trim();
      if (!still || still !== lastQuery){ hideGhost(); return; }

      $ghostPortal.html(renderGhostHTML(suggestions))
        .off('click', '.ai-suggest-item')
        .on('click', '.ai-suggest-item', function(){
          $ta.val($(this).text().trim()).trigger('input');
          $ta.trigger('textComplete:hide');
          hideGhost();
        })
        .stop(true,true).fadeIn(120);

      placeGhostAboveInput();
    }catch(e){
      console.log('AI suggest error:', e);
      $ghostPortal.find('.text-secondary').text('Could not load suggestions');
      placeGhostAboveInput();
    }
  }

  $ta.on('input', function(){
    const v = (this.value || '').trim();
    setSendDisabled(!v);
    clearTimeout(aiTimer);

    if (!v){ hideGhost(); return; }

    if (v.length >= AI_SUGGEST_MIN_LEN && v.length <= AI_SUGGEST_MAX_LEN){
      aiTimer = setTimeout(()=>fetchAISuggestions(v), 500);
    } else {
      hideGhost();
    }
  });

  /* =================== Layout (no overlap) =================== */
  function placeDropdownBelowInput(){
    const $dd = $('.textcomplete-dropdown:visible');
    if (!$dd.length || !$wrapper.length){ setInputLift(0); return; }

    $dd.css({ position:'fixed', visibility:'hidden', display:'block', height:'', maxHeight:'360px' });

    const r   = getWrapRect();
    const gap = 8;

    // measure dropdown
    const measuredH = $dd.outerHeight();
    const spaceBelow = window.innerHeight - (r.bottom + gap);

    // how far the input should lift to fit the dropdown fully
    const liftNeeded = Math.max(0, measuredH - spaceBelow + 8);
    const maxLift    = Math.max(0, r.top - 16);

    // if ghost is visible, don't collide into it
    const gp = $('#ghost-portal:visible')[0];
    const ghostRect = gp ? gp.getBoundingClientRect() : null;
    const maxLiftByGhost = ghostRect ? Math.max(0, (r.top - MIN_GAP) - ghostRect.bottom) : maxLift;

    const lift = Math.min(liftNeeded, maxLift, maxLiftByGhost);
    setInputLift(lift);

    const top = r.bottom - lift + gap;

    // cap dropdown height to remaining viewport
    const finalMaxH = Math.min(360, Math.max(160, window.innerHeight - top - 8));
    $dd.css({ top, left:r.left, width:r.width + 'px', maxHeight: finalMaxH + 'px', visibility:'visible', zIndex: Z_DROPDOWN });
  }

  function layoutOverlays(){ placeDropdownBelowInput(); placeGhostAboveInput(); }
  function layoutOverlaysNextPaint(){ requestAnimationFrame(()=>requestAnimationFrame(layoutOverlays)); }

  $ta.on('textComplete:show textComplete:rendered textComplete:append textComplete:hide', layoutOverlaysNextPaint);
  $(window).on('resize scroll', layoutOverlaysNextPaint);
  $('.chat-content').on('scroll', layoutOverlaysNextPaint);

  $(document).on('click', (e)=>{
    if (!$(e.target).closest('#ghost-portal, .input-wrapper, .textcomplete-dropdown').length) hideGhost();
  });
  $ta.on('blur', hideGhost);

  const observer = new MutationObserver(()=>{ if (!$('.textcomplete-dropdown:visible').length) setInputLift(0); });
  observer.observe(document.body, { childList:true, subtree:true });

  $(function(){ hideGhost(); setSendDisabled(true); });

  /* =================== Quick-match strategies =================== */

  // Territory: T\d+
  (function territoryStrategy(){
    let term='', page=1, more=false;
    $ta.textcomplete([{
      match: /(?:^|[\s,])([Tt]\d{1,})$/, index:1, cache:false,
      search: function(t, callback){
        const ec = explicitContext(); if (ec && ec.label && ec.label!=='Territory'){ callback([]); return; }
        term = t.trim(); page = 1;
        const epoch = ++__epoch;
        ajaxFetch('Territory', term, page, ({items, more:m})=>{
          if (epoch !== __epoch) return;
          more = m; callback(items);

          requestAnimationFrame(()=>{
            const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
            attachInfiniteScroll(dd, ()=>{
              if (epoch !== __epoch || !more) return;
              ajaxFetch('Territory', term, ++page, ({items:next, more:m2})=>{
                if (epoch !== __epoch) return;
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x=>{
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              }, 'territory', epoch);
            }, 'infTerritory');
          });
        }, 'territory', epoch);
      },
      replace: function(item){
        const result = specialReplace('Territory', item);
        $ta.val(result).trigger('input'); $ta.trigger('textComplete:hide');
      },
      template: (item)=>`
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">Territory</small>
            </div>
          </div>
        </div>`
    }], { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });
  })();

  // Sales Zone: Z\d+
  (function szoneStrategy(){
    let term='', page=1, more=false;
    $ta.textcomplete([{
      match: /(?:^|[\s,])([Zz]\d{1,})$/, index:1, cache:false,
      search: function(t, callback){
        const ec = explicitContext(); if (ec && ec.label && ec.label!=='Sales Zone'){ callback([]); return; }
        term = t.trim(); page = 1;
        const epoch = ++__epoch;
        ajaxFetch('Sales Zone', term, page, ({items, more:m})=>{
          if (epoch !== __epoch) return;
          more = m; callback(items);

          requestAnimationFrame(()=>{
            const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
            attachInfiniteScroll(dd, ()=>{
              if (epoch !== __epoch || !more) return;
              ajaxFetch('Sales Zone', term, ++page, ({items:next, more:m2})=>{
                if (epoch !== __epoch) return;
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x=>{
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              }, 'szone', epoch);
            }, 'infSzone');
          });
        }, 'szone', epoch);
      },
      replace: function(item){
        const result = specialReplace('Sales Zone', item);
        $ta.val(result).trigger('input'); $ta.trigger('textComplete:hide');
      },
      template: (item)=>`
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">Sales Zone</small>
            </div>
          </div>
        </div>`
    }], { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });
  })();

  // Material Group
  (function matklStrategy(){
    let term='', page=1, more=false;
    const rx = /(?:^|[\s,])((?:[Ff]\d+|[Bb][Pp]\d+|[Mm][Vv]\d+|[Bb]\d+|[Ss](?:[Oo])?\d+|\d{3,6}))$/;

    $ta.textcomplete([{
      match: rx, index:1, cache:false,
      search: function(t, callback){
        const ec = explicitContext(); if (ec && ec.label && ec.label!=='Material Group'){ callback([]); return; }
        term = t.trim(); page = 1;
        const epoch = ++__epoch;
        ajaxFetch('Material Group', term, page, ({items, more:m})=>{
          if (epoch !== __epoch) return;
          more = m; callback(items);

          requestAnimationFrame(()=>{
            const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
            attachInfiniteScroll(dd, ()=>{
              if (epoch !== __epoch || !more) return;
              ajaxFetch('Material Group', term, ++page, ({items:next, more:m2})=>{
                if (epoch !== __epoch) return;
                more = m2;
                const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x=>{
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), x.meta)}</a>`;
                  ul.appendChild(li);
                });
              }, 'matkl', epoch);
            }, 'infMatkl');
          });
        }, 'matkl', epoch);
      },
      replace: function(item){
        const result = specialReplace('Material Group', item);
        $ta.val(result).trigger('input'); $ta.trigger('textComplete:hide');
      },
      template: (item)=>`
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">Material Group</small>
            </div>
          </div>
        </div>`
    }], { maxCount: 15, debounce: 120, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });
  })();

  // Division Names & Codes
  (function divisionStrategy(){
    let term='', page=1, more=false;
    const DIVISION_NAMES = [
      'Decorative','Industrial Paints','Adhesive & Chemicals','Powder Coating','Wood Coating','Marine Paints',
      'Vehicle Refinish','Printing Ink','Trading','Texbond','Service','Construction Chemica','Food Grade Can',
      'Industrial Grade Can','Container Scraps','Coil Coating','Training Service'
    ];
    const DIVISION_CODES = new Set(['10','20','50','40','80','30','70','11','60','55','65','90','3','2','4','95','66']);

    // Names
    $ta.textcomplete([{
      match: /(?:^|[\s,])([A-Za-z][A-Za-z\s]{3,})$/, index:1, cache:false,
      search: function(t, callback){
        const ec = explicitContext(); if (ec && ec.label){ callback([]); return; }

        term = (t||'').trim();
        if (term.length < 4){ callback([]); return; }

        const lower = term.toLowerCase();
        const local = DIVISION_NAMES.filter(n=>n.toLowerCase().startsWith(lower))
          .slice(0,6).map(n=>({ id:n, text:n, meta:'Division' }));
        callback(local);

        page = 1;
        ajaxFetch('Division', term, page, ({items, more:m})=>{
          more = m;
          const seen = new Set(local.map(i=>i.text.toLowerCase()));
          const merged = local.concat(items.filter(i=>!seen.has(i.text.toLowerCase())));
          callback(merged);

          requestAnimationFrame(()=>{
            const dd = document.querySelector('.textcomplete-dropdown'); if (!dd) return;
            attachInfiniteScroll(dd, ()=>{
              if (!more) return;
              ajaxFetch('Division', term, ++page, ({items:next, more:m2})=>{
                more = m2; const ul = dd.querySelector('ul'); if (!ul) return;
                next.forEach(x=>{
                  const key = x.text.toLowerCase(); if (seen.has(key)) return; seen.add(key);
                  const li = document.createElement('li');
                  li.className = 'textcomplete-item';
                  li.innerHTML = `<a>${cardHTML(hl(x.text, term), 'Division')}</a>`;
                  ul.appendChild(li);
                });
              }, 'division');
            }, 'infDivName');
          });
        }, 'division');
      },
      replace: function(item){ const r = specialReplace('Division', item); $ta.val(r).trigger('input'); $ta.trigger('textComplete:hide'); },
      template: (item)=>`
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex align-items-center">
              <div class="suggest-title">${hl(item.text, term)}</div>
              <small class="suggest-meta ms-auto">Division</small>
            </div>
          </div>
        </div>`
    }], { maxCount: 15, debounce: 140, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });

    // Codes
    $ta.textcomplete([{
      match: /(?:^|[\s,])(\d{1,2})$/, index:1, cache:false,
      search: function(t, callback){
        const ec = explicitContext(); if (ec && ec.label && ec.label!=='Division Code'){ callback([]); return; }
        const code = (t||'').trim();
        if (DIVISION_CODES.has(code)){ callback([{ id:code, text:code, meta:'Division Code' }]); }
        else { callback([]); }
      },
      replace: function(item){ const r = specialReplace('Division Code', item); $ta.val(r).trigger('input'); $ta.trigger('textComplete:hide'); },
      template: (item)=>cardHTML(item.text, 'Division Code')
    }], { maxCount: 8, debounce: 80, zIndex: 10000, dropdownClassName: 'textcomplete-dropdown' });
  })();

})();
