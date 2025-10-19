
(function () {
   let setConversationId = null; 
  const INPUT_SELECTOR = '#message-input';
  const SEND_BTN_SELECTOR = '#send-btn';
  const AUTOCOMPLETE_URL = '/api/sales/autocomplete/';
  const AI_SUGGEST_URL = '/api/sales/prompt-suggestions/';
  const JWT_TOKEN = localStorage.getItem("auth_token");
  if (JWT_TOKEN) $.ajaxSetup({ headers: { 'Authorization': 'Bearer ' + JWT_TOKEN } });


  /* ---------------- Labels & regex ---------------- */
  const FIELD_LABELS = [
    "Dealer","Brand","Product Name","Material Group","Division","Company Code","Sales Org",
    "Distribution Channel","Business Area","Credit Control Area","Dealer Group","Account Group",
    "Sales Group","Sales Office","Payer ID","Product Code","Volume Unit","Business Group",
    "Territory","Sales Zone","Date","Dealer Code","Invoice Number"
  ];
  const fieldAlt = FIELD_LABELS.slice().sort((a,b)=>b.length-a.length)
    .map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|');
  const findAnyFieldContext = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*([^,:]*)$`,'i');
  const findNewFieldStart  = new RegExp(`(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*$`,'i');

  /* ---------------- Elements ---------------- */
  const $ta        = $(INPUT_SELECTOR);
  const $wrapper   = $('.input-wrapper');     // wraps textarea + send button
  const $inputArea = $('.input-area');        // we animate this a bit when needed

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
  $inputArea.css({ transition: 'transform 160ms ease', willChange: 'transform' });

  /* ---------------- Helpers ---------------- */
  function normalizeFieldName(field){
    const map = {
      "dealer":"Dealer","brand":"Brand","product name":"Product Name","material group":"Material Group",
      "division":"Division","company code":"Company Code","sales org":"Sales Org",
      "distribution channel":"Distribution Channel","business area":"Business Area",
      "credit control area":"Credit Control Area","dealer group":"Dealer Group","account group":"Account Group",
      "sales group":"Sales Group","sales office":"Sales Office","payer id":"Payer ID","product code":"Product Code",
      "volume unit":"Volume Unit","business group":"Business Group","territory":"Territory",
      "sales zone":"Sales Zone","date":"Date","dealer code":"Dealer Code","invoice number":"Invoice Number"
    };
    return map[field?.toLowerCase()] || field;
  }
  function highlight(t,term){
    if(!term) return t;
    try { return t.replace(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`,'ig'), '<span class="hl">$1</span>'); }
    catch { return t; }
  }
  function setSendDisabled(v){ $(SEND_BTN_SELECTOR).prop('disabled', !!v); }
  function getWrapRect(){ return $wrapper[0].getBoundingClientRect(); }
  function hideGhost(){ $ghostPortal.stop(true,true).fadeOut(120); }
  function setInputLift(px){ $inputArea.css('transform', `translateY(${-Math.max(0, px)}px)`); }

  /* ---------------- Smart suggestions (textcomplete) ---------------- */
  let curField=null, curTerm='', curPage=1, hasMore=false, currentRequest=null;

  function searchField(field, q, page, cb){
    $.getJSON(AUTOCOMPLETE_URL, { field: normalizeFieldName(field), q, page })
      .done(resp=>{
        const items=(resp.results||[]).map(r=>({ id:r.id, text:r.text, meta:r.meta||null }));
        cb({ items, more: !!(resp.pagination && resp.pagination.more) });
      })
      .fail(()=>cb({ items:[], more:false }));
  }
  function attachInfiniteScroll(dropdownEl, loader){
    const $dd=$(dropdownEl);
    $dd.off('scroll.inf').on('scroll.inf', function(){
      if(this.scrollTop + this.clientHeight + 8 >= this.scrollHeight) loader();
    });
  }

  $ta.textcomplete([{
    match:/([^\s,].*)$/,
    index:1,
    cache:false,
    search:function(term, callback){
      if(currentRequest){ try{ currentRequest.abort(); }catch{} }
      const text=$ta.val().slice(0,$ta[0].selectionStart);

      const ctx=text.match(findAnyFieldContext);
      if(ctx){
        curField=ctx[1].trim(); curTerm=(ctx[2]||'').trim(); curPage=1;
        if(curTerm.length>=1){
          currentRequest = searchField(curField, curTerm, curPage, ({items,more})=>{
            hasMore=more; callback(items);
            requestAnimationFrame(()=>{
              const dd=document.querySelector('.textcomplete-dropdown');
              if(!dd) return;
              attachInfiniteScroll(dd, ()=>{
                if(!hasMore) return;
                searchField(curField, curTerm, ++curPage, ({items:next, more:m2})=>{
                  hasMore=m2; const ul=dd.querySelector('ul'); if(!ul) return;
                  next.forEach(x=>{
                    const li=document.createElement('li'); li.className='textcomplete-item';
                    li.innerHTML=`
                      <a><div class="card suggest-card"><div class="card-body py-2 px-3">
                        <div class="d-flex justify-content-between align-items-center gap-2">
                          <div class="suggest-title">${highlight(x.text, curTerm)}</div>
                          ${x.meta?`<small class="suggest-meta text-truncate">${x.meta}</small>`:''}
                        </div>
                      </div></div></a>`;
                    ul.appendChild(li);
                  });
                  layoutOverlaysNextPaint();
                });
              });
            });
            layoutOverlaysNextPaint();
          });
          return;
        }
      }

      const nfs=text.match(findNewFieldStart);
      if(nfs){ curField=nfs[1].trim(); curTerm=''; return callback([]); }

      const lastComma=text.lastIndexOf(',');
      if(lastComma!==-1){
        const before=text.slice(0,lastComma);
        const fm=before.match(new RegExp(`\\b(${fieldAlt})\\s*:`, 'i'));
        if(fm){
          curField=fm[1]; curTerm=text.slice(lastComma+1).trim(); curPage=1;
          if(curTerm.length>=1){
            currentRequest=searchField(curField, curTerm, curPage, ({items,more})=>{
              hasMore=more; callback(items); layoutOverlaysNextPaint();
            });
            return;
          }
        }
      }

      curField=null; curTerm=''; callback([]);
    },
    replace:function(item){
      const full=$ta.val(), pos=$ta[0].selectionStart;
      const before=full.slice(0,pos), after=full.slice(pos);
      const label=curField?curField.trim():'';
      const labelRx=new RegExp(`\\b${label}\\b\\s*:?`,'i');

      const ctx=before.match(findAnyFieldContext);
      if(ctx){
        const fieldValue=(ctx[2]||'').trim();
        const start=before.lastIndexOf(fieldValue);
        return `${full.slice(0,start)}${item.text}${full.slice(pos)}`;
      }
      const lastComma=before.lastIndexOf(',');
      if(lastComma!==-1){
        const b=full.slice(0,lastComma+1), a=full.slice(pos);
        return (label && !b.match(labelRx)) ? `${b} ${label} ${item.text}, ${a}` : `${b} ${item.text}, ${a}`;
      }
      const nfs=before.match(findNewFieldStart);
      if(nfs){
        const fname=nfs[1].trim(), pref=before.endsWith(',')?'':', ';
        return `${full.slice(0,pos)}${pref}${fname} ${item.text}, ${after}`;
      }
      const pref=before.trim().endsWith(',')?' ':', ';
      return label?`${before}${pref}${label} ${item.text}, ${after}`:`${before}${pref}${item.text}, ${after}`;
    },
    template:function(item){
      return `<div class="card suggest-card"><div class="card-body py-2 px-3">
                <div class="d-flex justify-content-between align-items-center gap-2">
                  <div class="suggest-title">${highlight(item.text,curTerm)}</div>
                  ${item.meta?`<small class="suggest-meta text-truncate">${item.meta}</small>`:''}
                </div>
              </div></div>`;
    }
  }],
  { maxCount:15, debounce:300, zIndex:10000, dropdownClassName:'textcomplete-dropdown' });

  /* ---------------- Ghost suggestions (ABOVE input, via body portal) ---------------- */
  let aiTimer=null, lastQuery='';

  function fallbackGhost(text){
    const key=(text||'').trim().split(/\s+/).slice(-2).join(' ') || 'this';
    return [
      `Show me the sales trend for ${key} over the past year.`,
      `Compare ${key}'s sales with others in the same region.`,
      `Break down ${key}'s sales by product category.`
    ];
  }
  function renderGhostHTML(sugs){
    return `<div class="card ai-suggest-card shadow-sm">
      <div class="card-body py-2 px-3">
        <div class="fw-semibold mb-1 text-secondary small">AI Suggestions</div>
        ${sugs.map(s=>`<div class="ai-suggest-item py-1 px-2 rounded-2 mb-1" style="cursor:pointer;">${s}</div>`).join('')}
      </div></div>`;
  }
  function placeGhostAboveInput(){
    if(!$ghostPortal.is(':visible')) return;

    // Make measurable
    $ghostPortal.css({ visibility:'hidden', display:'block' });

    const r = getWrapRect();
    const left = r.left;
    const width= r.width;

    const viewportTop = 8;
    const anchorBottom = r.top - 10; // 10px above input
    const maxH = Math.max(120, anchorBottom - viewportTop);

    $ghostPortal.find('.ai-suggest-card').css({ maxHeight: maxH + 'px', overflow:'auto' });

    const gh  = $ghostPortal.outerHeight();
    const top = Math.max(viewportTop, anchorBottom - gh);

    $ghostPortal.css({ top, left, width, visibility:'visible' });
  }

  async function fetchAISuggestions(text){
     const getConversationId = $('#chat-id-holder').data('current-conversation-id');
 const chatId = $('#chat-id-holder').data('current-conversation-id');
console.log("Chat ID:", chatId);

console.log("currentChatId on init  ", getConversationId); 
    console.log(chatId)
    const q=(text||'').trim();
    if(!q){ hideGhost(); return; }
    lastQuery=q;

    let suggestions=[];
    try{
      const resp=await $.ajax({
        url: AI_SUGGEST_URL, method:'POST',
        data: JSON.stringify({ input_text:q, conversation_id: getConversationId }),
        contentType:'application/json'
      });
      suggestions=(resp && resp.suggestions) || [];
    }catch(e){
      // ignore, we'll use fallback
    }
    if(!suggestions.length) suggestions=fallbackGhost(q);

    // guard against stale input
    const still=($ta.val()||'').trim();
    if(!still || still!==lastQuery){ hideGhost(); return; }

    $ghostPortal
      .html(renderGhostHTML(suggestions))
      .off('click', '.ai-suggest-item')
      .on('click', '.ai-suggest-item', function(){
        $ta.val($(this).text().trim()).trigger('input'); hideGhost();
      })
      .stop(true,true).fadeIn(120);

    placeGhostAboveInput();
  }

  // Input trigger (6+ chars, debounced)
  $ta.on('input', function(){
    const v=(this.value||'').trim();
    setSendDisabled(!v);
    clearTimeout(aiTimer);

    if(!v){ hideGhost(); return; }

    if(v.length >= 6){
      aiTimer=setTimeout(()=>fetchAISuggestions(v), 500);
    }else{
      hideGhost();
    }
  });

  /* ---------------- Layout: smart below (with lift), ghost above ---------------- */
  function placeDropdownBelowInput(){
    const $dd=$('.textcomplete-dropdown:visible');
    if(!$dd.length || !$wrapper.length){
      setInputLift(0);
      return;
    }
    $dd.css({ position:'fixed', visibility:'hidden', display:'block', height:'', maxHeight:'360px' });

    const r   = getWrapRect();
    const ddH = $dd.outerHeight();
    const gap = 8;

    const spaceBelow = window.innerHeight - (r.bottom + gap);
    const liftNeeded = Math.max(0, ddH - spaceBelow + 8);
    const maxLift    = Math.max(0, r.top - 16);
    const lift       = Math.min(liftNeeded, maxLift);

    setInputLift(lift);
    const top = r.bottom - lift + gap;

    $dd.css({ top, left:r.left, width:r.width+'px', visibility:'visible', zIndex:10010 });
  }

  function layoutOverlays(){
    placeDropdownBelowInput();  // smart suggestions below (with lift)
    placeGhostAboveInput();     // ghost above input (portal)
  }
  function layoutOverlaysNextPaint(){ requestAnimationFrame(()=>requestAnimationFrame(layoutOverlays)); }

  $ta.on('textComplete:show textComplete:rendered textComplete:append textComplete:hide', layoutOverlaysNextPaint);
  $(window).on('resize scroll', layoutOverlaysNextPaint);
  $('.chat-content').on('scroll', layoutOverlaysNextPaint);

  // Hide ghost on blur / outside click
  $(document).on('click', (e)=>{
    if(!$(e.target).closest('#ghost-portal, .input-wrapper, .textcomplete-dropdown').length) hideGhost();
  });
  $ta.on('blur', hideGhost);

  // Reset lift if dropdown hides
  const observer=new MutationObserver(()=>{ if (!$('.textcomplete-dropdown:visible').length) setInputLift(0); });
  observer.observe(document.body, { childList:true, subtree:true });

  // Init
  $(function(){
    hideGhost();
    setSendDisabled(true);
  });
})();

