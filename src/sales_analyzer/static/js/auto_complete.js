(function () {
  'use strict';
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

  $ghostPortal.css({
    position: 'fixed',
    zIndex: Z_GHOST,
    display: 'none',
    pointerEvents: 'auto'
  });
  $inputArea.css({
    transition: 'transform 160ms ease, opacity 160ms ease',
    willChange: 'transform'
  });

  let __epoch = 0;

  /* =================== Helpers =================== */
  const FIELD_LABELS = [
    'Dealer','Brand','Product Name','Material Group','Division','Company Code','Sales Org',
    'Distribution Channel','Business Area','Credit Control Area','Dealer Group','Account Group',
    'Sales Group','Sales Office','Payer ID','Product Code','Volume Unit','Business Group',
    'Territory','Sales Zone','Date','Dealer Code','Invoice Number'
  ];
  const fieldAlt = FIELD_LABELS.slice().sort((a,b)=>b.length-a.length)
    .map((s)=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|');

  const PREP_WORDS = ['from','of','in','by','within','for'];

  const DIVISION_NAMES = [
    'Decorative','Industrial Paints','Adhesive & Chemicals','Powder Coating','Wood Coating','Marine Paints',
    'Vehicle Refinish','Printing Ink','Trading','Texbond','Service','Construction Chemica','Food Grade Can',
    'Industrial Grade Can','Container Scraps','Coil Coating','Training Service'
  ];

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

  function escapeRx(s){ return String(s).replace(/[.*+?^${}()|[\]\\]/g,'\\$&'); }

  function hl(t, term){
    if (!term) return t;
    try {
      return t.replace(new RegExp(`(${escapeRx(term)})`,'ig'), '<span class="hl">$1</span>');
    } catch {
      return t;
    }
  }

  function setSendDisabled(v){ $(SEND_BTN_SELECTOR).prop('disabled', !!v); }

  function getWrapRect(){
    return $wrapper[0]?.getBoundingClientRect?.() || { top:0,left:0,width:0,bottom:0 };
  }

  function setInputLift(px){
    $inputArea.css('transform', `translateY(${-Math.max(0,px)}px)`);
  }

  function hideGhost(){
    $ghostPortal.stop(true,true).fadeOut(120);
  }

  // Keep previous selections in the textarea, merge by label
  function insertLabeled(label, value){
    const $i = $ta;
    let full = $i.val();
    const pos = $i[0].selectionStart;
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

    const groups = {};
    const order = [];
    for (const p of parts){
      const m = p.match(new RegExp(`^(${fieldAlt})\\s+(.+)$`,'i'));
      if (!m) continue;
      const lbl = canonLabel(m[1]);
      const val = squish(m[2]);
      if (!groups[lbl]){ groups[lbl]=[]; order.push(lbl); }
      if (!groups[lbl].includes(val)) groups[lbl].push(val);
    }

    if (!groups[L]){ groups[L]=[V]; order.push(L); }
    else {
      if (MULTI.has(L)){
        if (!groups[L].includes(V)) groups[L].push(V);
      } else {
        groups[L]=[V];
      }
    }

    let out = '';
    for (const lbl of order){
      if (groups[lbl]?.length) out += `${lbl} ${groups[lbl].join(', ')}, `;
    }
    return out + (after ? after + ' ' : '');
  }

  function specialReplace(label, item){
    const L = normalizeFieldName(String(label||'').trim());
    const pickRaw = (x)=>!x ? '' : (typeof x === 'string'
      ? x
      : (x.commit || x.text || x.value || x.id || ''));
    let commit = pickRaw(item);

    if (/^Dealer$/i.test(L)){
      commit = (commit||'')
        .replace(/\s*[-–—|]\s.*$/, '')
        .replace(/,\s*$/, '')
        .trim();
    }

    const $i = $ta;
    let full = $i.val();
    const pos = $i[0].selectionStart;
    const left  = full.slice(0, pos);
    const right = full.slice(pos);
    const rx = new RegExp(`(^|[\\s,])\\s*(${escapeRx(L)})\\s*:?\\s*([^,\\n]*)$`, 'i');
    const m  = rx.exec(left);

    if (m){
      const prefix = left.slice(0, m.index);
      const delim  = m[1] || ' ';
      const replacedLeft = `${prefix}${delim}${L} ${String(commit||'').trim()}, `;
      full = replacedLeft + right.replace(/^(\s*),\s*/, '$1');
      return full;
    }
    return insertLabeled(L, String(commit||'').trim());
  }

  /* =================== Field/Term detection =================== */

  const LEADING_FILLER_RX = new RegExp(
    '^\\s*(?:list|who\\s+(?:is|are|bought|buy|purchased|has|have)|who|that|which|with|having|bought|buy|purchased|has|have|of|from|in|by|within|for|and)\\s+',
    'i'
  );

  function cleanLeadingFiller(s){
    let out = String(s || '');
    for (let i=0;i<4;i++){
      const prev = out;
      out = out.replace(LEADING_FILLER_RX, '');
      if (out === prev) break;
    }
    return out.trim();
  }

  function startsWithAnyIgnoreCase(term, arr) {
    const t = (term || '').trim().toLowerCase();
    if (t.length < 3) return false;
    for (const n of arr) if ((n||'').toLowerCase().startsWith(t)) return true;
    return false;
  }

  // Quick tail detectors (codes / division prefix)
  function detectQuickTail(text){
    const tail = text;
    let m = tail.match(/(?:^|[\s,])(Z\d{1,})$/i);
    if (m) return { field:'Sales Zone', q:m[1].toUpperCase() };

    m = tail.match(/(?:^|[\s,])(T\d{1,})$/i);
    if (m) return { field:'Territory', q:m[1].toUpperCase() };

    m = tail.match(/(?:^|[\s,])((?:[Ff]\d+|[Bb][Pp]\d+|[Mm][Vv]\d+|[Bb]\d+|[Ss](?:[Oo])?\d+|\d{3,6}))$/);
    if (m) return { field:'Material Group', q:m[1] };

    const w = (tail.match(/([A-Za-z][A-Za-z\s]{2,})$/)||[])[1] || '';
    if (startsWithAnyIgnoreCase(w, DIVISION_NAMES)) return { field:'Division', q:w.trim() };

    return null;
  }

  function findLastFieldMention(text){
    let best = null;
    for (const name of FIELD_LABELS){
      const rx = new RegExp(`\\b${escapeRx(name)}\\b`, 'ig');
      let m;
      while ((m = rx.exec(text))){
        best = { field: name, index: m.index, afterIdx: m.index + m[0].length };
      }
    }
    return best;
  }

  function extractTermFrom(text, startIdx){
    let s = text.slice(startIdx).replace(/^\s*:?\s*/, '');

    let cutAt = s.length;
    for (const name of FIELD_LABELS){
      const rx = new RegExp(`\\b${escapeRx(name)}\\b`, 'i');
      const m = rx.exec(s);
      if (m && m.index < cutAt) cutAt = m.index;
    }

    const commaIdx = s.search(/[,\n]/);
    if (commaIdx >= 0 && commaIdx < cutAt) cutAt = commaIdx;

    // If no cut point was found (cutAt == full length), the text ran to end with no field label
    // or comma terminator — likely a natural-language sentence fragment, not an entity name.
    // Limit to first 5 words; if the first word is a common filler/verb, suppress entirely.
    if (cutAt === s.length) {
      const words = s.trim().split(/\s+/).filter(Boolean);
      if (words.length > 5) return '';           // sentence fragment → suppress
    }

    s = s.slice(0, cutAt).trim();

    s = cleanLeadingFiller(s);

    return s.trim();
  }

  function parseActiveField(textLeft){
    // 1) quick tail wins
    const quick = detectQuickTail(textLeft);
    if (quick) return quick;

    // 2) last explicit label anywhere
    const lastLabel = findLastFieldMention(textLeft);

    // 3) last preposition clause naming a field
    let pref = null;
    const prepClauseRx = new RegExp(
      `\\b(?:${PREP_WORDS.join('|')})\\b\\s+(${fieldAlt})\\b\\s*:?\\s*([^,\\n]*)`,
      'ig'
    );
    let m;
    while ((m = prepClauseRx.exec(textLeft))){
      pref = {
        field: normalizeFieldName(m[1]),
        q: cleanLeadingFiller(m[2] || '').trim(),
        idx: m.index
      };
    }

    if (pref && pref.q){
      // if a later label exists (e.g. "Brand" or "Division"), prefer that
      if (lastLabel && lastLabel.index > pref.idx && lastLabel.field !== pref.field){
        const q2 = extractTermFrom(textLeft, lastLabel.afterIdx);
        if (q2) return { field: lastLabel.field, q: q2 };
      }
      return { field: pref.field, q: pref.q };
    }

    // 4) fallback to last explicit label
    if (lastLabel){
      const q = extractTermFrom(textLeft, lastLabel.afterIdx);
      return { field: lastLabel.field, q };
    }

    return null;
  }

  /* =================== Autocomplete lifecycle =================== */

  let curField = null, curTerm = '', curPage = 1, hasMore = false, currentRequest = null;
  // Tracks fields whose last API call returned no results for a given q prefix.
  // Key: field name, Value: the q string that returned empty.
  // If current q starts with the cached empty q → skip the API call.
  const emptyResultCache = new Map();

  function hardResetAutocomplete(){
    try { if (currentRequest && currentRequest.abort) currentRequest.abort(); } catch {}
    currentRequest = null;
    __epoch++;
    hasMore = false;
    $('.textcomplete-dropdown').hide().empty();
    setInputLift(0);
  }

  function ensureDropdown(){
    let $dd = $('.textcomplete-dropdown');
    if (!$dd.length) $dd = $('<ul class="textcomplete-dropdown"></ul>').appendTo('body');
    $dd.css({ zIndex: Z_DROPDOWN, position:'fixed', display:'block' });
    return $dd;
  }

  function startAutocomplete(field, q){
    if (!q) {
      $('.textcomplete-dropdown').hide().empty();
      return;
    }
    curField = field;
    curTerm = q;
    curPage = 1;

    const epoch = ++__epoch;
    const $dd = ensureDropdown();
    $dd.empty();

    const render = (items)=>{
      items.forEach(it=>{
        const li = $(`
          <li class="textcomplete-item">
            <a>
              <div class="card suggest-card">
                <div class="card-body py-2 px-3 d-flex justify-content-between align-items-center">
                  <div class="suggest-title">${hl(it.text || it.id, curTerm)}</div>
                  <small class="suggest-meta">${field}</small>
                </div>
              </div>
            </a>
          </li>
        `);
        li.on('mousedown', (e)=>{ e.preventDefault(); e.stopPropagation(); });
        li.on('click', ()=>{
          const result = specialReplace(field, it);
          $ta.val(result).trigger('input');
          $('.textcomplete-dropdown').hide();
        });
        $dd.append(li);
      });
      layoutOverlaysNextPaint();
    };

    currentRequest = $.getJSON(AUTOCOMPLETE_URL, { field, q, page: curPage })
      .done((resp)=>{
        if (epoch !== __epoch) return;
        const items = (resp.results || []).map((r)=>({
          id:r.id,
          text:r.text,
          meta:r.meta || null,
          commit:r.commit || r.insert || r.value || r.id || null
        }));
        if (items.length === 0) {
          // Only mark as exhausted once the query reaches 5+ characters with no results.
          // Below that threshold keep retrying — a longer prefix might still match.
          if (q.length >= 5) {
            emptyResultCache.set(field, q);
          }
          $('.textcomplete-dropdown').hide().empty();
        } else {
          emptyResultCache.delete(field);   // results found → clear any stale empty flag
          render(items);
        }
        hasMore = !!(resp.pagination && resp.pagination.more);

        requestAnimationFrame(()=>{
          const dd = document.querySelector('.textcomplete-dropdown');
          if (!dd) return;
          dd.onscroll = function(){
            if (this.scrollTop + this.clientHeight + 8 >= this.scrollHeight) {
              if (!hasMore || epoch !== __epoch) return;
              $.getJSON(AUTOCOMPLETE_URL, { field, q, page: ++curPage })
                .done((r2)=>{
                  if (epoch !== __epoch) return;
                  const next = (r2.results || []).map((r)=>({
                    id:r.id,
                    text:r.text,
                    meta:r.meta || null,
                    commit:r.commit || r.insert || r.value || r.id || null
                  }));
                  hasMore = !!(r2.pagination && r2.pagination.more);
                  render(next);
                })
                .fail(()=>{ hasMore = false; });
            }
          };
        });
      })
      .fail(()=>{
        if (epoch === __epoch) {
          $('.textcomplete-dropdown').hide().empty();
        }
      });
  }

  /* =================== Input handler: Autocomplete =================== */

  $ta.on('input', function(){
    const v = (this.value || '');
    setSendDisabled(!v.trim());

    if (!v.trim()) { hideGhost(); }

    const caret = this.selectionStart || 0;
    const prevChar = caret > 0 ? v.charAt(caret - 1) : '';

    // If user just typed a comma => end this filter chunk, full reset
    if (prevChar === ',') {
      hardResetAutocomplete();
      emptyResultCache.clear();   // new filter segment — all fields get a fresh try
      curField = null;
      curTerm  = '';
      curPage  = 1;
      hasMore  = false;
      return;
    }

    const textLeft = v.slice(0, caret);
    const detected = parseActiveField(textLeft);
    if (!detected) return;

    const nextField = detected.field;
    const nextQ     = (detected.q || '').trim();

    // ignore very short junk
    if (!nextQ || nextQ.length < 2) return;

    // Suppress sentence fragments — entity names don't exceed 5 words.
    // "name which sale is high by the dealer" → 8 words → skip.
    if (nextQ.split(/\s+/).filter(Boolean).length > 5) return;

    // Switching to a new field → give it a fresh try (clear its empty cache entry)
    if (nextField !== curField) {
      emptyResultCache.delete(nextField);
    }

    // Skip if this field already returned empty for this exact q prefix
    const cachedEmpty = emptyResultCache.get(nextField);
    if (cachedEmpty && nextQ.toLowerCase().startsWith(cachedEmpty.toLowerCase())) return;

    // If switching field or query changed -> reset & start new call
    if (nextField !== curField || nextQ !== curTerm){
      hardResetAutocomplete();
      startAutocomplete(nextField, nextQ);
    }
  });

  /* =================== AI Suggestions (unchanged behaviour) =================== */

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
          ${sugs.map(s => `
            <div class="ai-suggest-item py-1 px-2 rounded-2 mb-1 text-truncate" title="${s}" style="cursor:pointer;">
              ${s}
            </div>`).join('')}
        </div>
      </div>`;
  }

  function placeGhostAboveInput(){
    if (!$ghostPortal.is(':visible')) return;
    $ghostPortal.css({ visibility:'hidden', display:'block' });
    const r = getWrapRect();
    const gh = $ghostPortal.outerHeight();
    const top = Math.max(8, r.top - MIN_GAP - gh);
    $ghostPortal.css({
      top,
      left:r.left,
      width:r.width,
      visibility:'visible',
      zIndex: Z_GHOST
    });
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
        url: AI_SUGGEST_URL,
        method: 'POST',
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
          $('.textcomplete-dropdown').hide();
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

  // Second input handler: only for AI suggestions
  $ta.on('input', function(){
    const v = (this.value || '').trim();
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

    $dd.css({
      position:'fixed',
      visibility:'hidden',
      display:'block',
      height:'',
      maxHeight:'360px'
    });

    const r   = getWrapRect();
    const gap = 8;

    const measuredH = $dd.outerHeight();
    const spaceBelow = window.innerHeight - (r.bottom + gap);
    const liftNeeded = Math.max(0, measuredH - spaceBelow + 8);
    const maxLift    = Math.max(0, r.top - 16);

    const gp = $('#ghost-portal:visible')[0];
    const ghostRect = gp ? gp.getBoundingClientRect() : null;
    const maxLiftByGhost = ghostRect
      ? Math.max(0, (r.top - MIN_GAP) - ghostRect.bottom)
      : maxLift;

    const lift = Math.min(liftNeeded, maxLift, maxLiftByGhost);
    setInputLift(lift);

    const top = r.bottom - lift + gap;
    const finalMaxH = Math.min(
      360,
      Math.max(160, window.innerHeight - top - 8)
    );
    $dd.css({
      top,
      left:r.left,
      width:r.width + 'px',
      maxHeight: finalMaxH + 'px',
      visibility:'visible',
      zIndex: Z_DROPDOWN
    });
  }

  function layoutOverlays(){
    placeDropdownBelowInput();
    placeGhostAboveInput();
  }

  function layoutOverlaysNextPaint(){
    requestAnimationFrame(()=>requestAnimationFrame(layoutOverlays));
  }

  $(window).on('resize scroll', layoutOverlaysNextPaint);
  $('.chat-content').on('scroll', layoutOverlaysNextPaint);

  $(document).on('click', (e)=>{
    if (!$(e.target).closest('#ghost-portal, .input-wrapper, .textcomplete-dropdown').length) {
      $('.textcomplete-dropdown').hide();
      hideGhost();
    }
  });

  // Singleton MutationObserver
  (function initSalesACObserver(){
    const key = '__salesACObserver';
    const w = window;
    try { if (w[key] && w[key].disconnect) w[key].disconnect(); } catch {}
    w[key] = new MutationObserver(() => {
      if (!$('.textcomplete-dropdown:visible').length) setInputLift(0);
    });
    w[key].observe(document.body, { childList: true, subtree: true });
  })();

  $(function(){
    hideGhost();
    setSendDisabled(true);
  });

})();
