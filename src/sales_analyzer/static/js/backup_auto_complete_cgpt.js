(function () {
  const INPUT_SELECTOR = '#message-input';
  const SEND_BTN_SELECTOR = '#send-btn';
  const AUTOCOMPLETE_URL = '/api/sales/autocomplete/';
  const PAGE_SIZE = 50;
  const JWT_TOKEN = localStorage.getItem("auth_token");

  if (JWT_TOKEN) $.ajaxSetup({ headers: { 'Authorization': 'Bearer ' + JWT_TOKEN } });

  const FIELD_LABELS = [
    "Dealer", "Brand", "Product Name", "Material Group", "Division", "Company Code", "Sales Org",
    "Distribution Channel", "Business Area", "Credit Control Area", "Dealer Group", "Account Group",
    "Sales Group", "Sales Office", "Payer ID", "Product Code", "Volume Unit", "Business Group",
    "Territory", "Sales Zone", "Date", "Dealer Code", "Invoice Number"
  ];

  // Build alternation for field labels
  const fieldAlt = FIELD_LABELS.slice().sort((a, b) => b.length - a.length)
    .map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');

  // Enhanced regex to find ANY field context (not just the last one)
  const findAnyFieldContext = new RegExp(
    `(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*([^,:]*)$`,
    'i'
  );

  // NEW: Detect if we're starting a new field
  const findNewFieldStart = new RegExp(
    `(?:^|[,\\s])\\s*(${fieldAlt})\\s*:?\\s*$`,
    'i'
  );

  const $ta = $(INPUT_SELECTOR);

  function searchField(field, q, page, cb) {
    console.log(`Searching ${field} for: "${q}", page: ${page}`);
    
    const normalizedField = normalizeFieldName(field);
    console.log(`Normalized field: "${normalizedField}"`);
    
    $.getJSON(AUTOCOMPLETE_URL, { field: normalizedField, q, page })
      .done(resp => {
        const items = (resp.results || []).map(r => ({ 
          id: r.id, 
          text: r.text, 
          meta: r.meta || null 
        }));
        cb({ items, more: !!(resp.pagination && resp.pagination.more) });
      })
      .fail((xhr, status, error) => {
        console.error('Autocomplete API error:', error);
        cb({ items: [], more: false });
      });
  }

  // Normalize field name to match SAP_FIELD_MAPPINGS keys
  function normalizeFieldName(field) {
    const fieldMap = {
      "dealer": "Dealer",
      "brand": "Brand", 
      "product name": "Product Name",
      "material group": "Material Group",
      "division": "Division",
      "company code": "Company Code",
      "sales org": "Sales Org",
      "distribution channel": "Distribution Channel",
      "business area": "Business Area",
      "credit control area": "Credit Control Area",
      "dealer group": "Dealer Group",
      "account group": "Account Group",
      "sales group": "Sales Group",
      "sales office": "Sales Office",
      "payer id": "Payer ID",
      "product code": "Product Code",
      "volume unit": "Volume Unit",
      "business group": "Business Group",
      "territory": "Territory",
      "sales zone": "Sales Zone",
      "date": "Date",
      "dealer code": "Dealer Code",
      "invoice number": "Invoice Number"
    };
    
    return fieldMap[field.toLowerCase()] || field;
  }

  function attachInfiniteScroll(dropdownEl, loader) {
    const $dd = $(dropdownEl);
    $dd.off('scroll.inf').on('scroll.inf', function () {
      if (this.scrollTop + this.clientHeight + 8 >= this.scrollHeight) loader();
    });
  }

  function highlight(text, term) {
    if (!term) return text;
    try {
      const rx = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig');
      return text.replace(rx, '<span class="hl">$1</span>');
    } catch { return text; }
  }

  // Paging state
  let curField = null, curTerm = '', curPage = 1, hasMore = false;
  let currentRequest = null;

  $ta.textcomplete([{
    match: /([^\s,].*)$/,
    index: 1,
    cache: false,
    search: function (term, callback) {
      // Cancel previous request if still pending
      if (currentRequest) {
        currentRequest.abort();
      }

      const text = $ta.val().slice(0, $ta[0].selectionStart);
      console.log('Searching in text:', text);
      
      // FIRST: Check if we're completing a field value (has field context)
      const fieldContext = text.match(findAnyFieldContext);
      if (fieldContext) {
        curField = fieldContext[1].trim();
        curTerm = (fieldContext[2] || '').trim();
        curPage = 1;

        console.log(`Field context detected - Field: "${curField}", Term: "${curTerm}"`);

        if (curTerm.length >= 1) {
          currentRequest = searchField(curField, curTerm, curPage, ({ items, more }) => {
            hasMore = more;
            callback(items);
            
            requestAnimationFrame(() => {
              const dd = document.querySelector('.textcomplete-dropdown');
              if (!dd) return;
              attachInfiniteScroll(dd, () => {
                if (!hasMore) return;
                searchField(curField, curTerm, ++curPage, ({ items: next, more: m2 }) => {
                  hasMore = m2;
                  const ul = dd.querySelector('ul');
                  if (!ul) return;
                  next.forEach(x => {
                    const li = document.createElement('li');
                    li.className = 'textcomplete-item';
                    li.innerHTML = `
                      <a>
                        <div class="card suggest-card">
                          <div class="card-body py-2 px-3">
                            <div class="d-flex justify-content-between align-items-center gap-2">
                              <div class="suggest-title">${highlight(x.text, curTerm)}</div>
                              ${x.meta ? `<small class="suggest-meta text-truncate">${x.meta}</small>` : ``}
                            </div>
                          </div>
                        </div>
                      </a>`;
                    ul.appendChild(li);
                  });
                });
              });
            });
          });
          return;
        }
      }
      
      // SECOND: Check if we're starting a NEW field (field name just typed)
      const newFieldStart = text.match(findNewFieldStart);
      if (newFieldStart) {
        curField = newFieldStart[1].trim();
        curTerm = '';
        console.log(`New field detected: "${curField}" - waiting for input`);
        return callback([]);
      }

      // THIRD: Check for multi-value context (after comma in same field)
      const lastComma = text.lastIndexOf(',');
      if (lastComma !== -1) {
        const textBeforeComma = text.slice(0, lastComma);
        const fieldMatch = textBeforeComma.match(new RegExp(`\\b(${fieldAlt})\\s*:`, 'i'));
        
        if (fieldMatch) {
          curField = fieldMatch[1];
          curTerm = text.slice(lastComma + 1).trim();
          curPage = 1;
          
          console.log(`Multi-value context - Field: "${curField}", Term: "${curTerm}"`);
          
          if (curTerm.length >= 1) {
            currentRequest = searchField(curField, curTerm, curPage, ({ items, more }) => {
              hasMore = more;
              callback(items);
              
              requestAnimationFrame(() => {
                const dd = document.querySelector('.textcomplete-dropdown');
                if (!dd) return;
                attachInfiniteScroll(dd, () => {
                  if (!hasMore) return;
                  searchField(curField, curTerm, ++curPage, ({ items: next, more: m2 }) => {
                    hasMore = m2;
                    const ul = dd.querySelector('ul');
                    if (!ul) return;
                    next.forEach(x => {
                      const li = document.createElement('li');
                      li.className = 'textcomplete-item';
                      li.innerHTML = `
                        <a>
                          <div class="card suggest-card">
                            <div class="card-body py-2 px-3">
                              <div class="d-flex justify-content-between align-items-center gap-2">
                                <div class="suggest-title">${highlight(x.text, curTerm)}</div>
                                ${x.meta ? `<small class="suggest-meta text-truncate">${x.meta}</small>` : ``}
                              </div>
                            </div>
                          </div>
                        </a>`;
                      ul.appendChild(li);
                    });
                  });
                });
              });
            });
            return;
          }
        }
      }

      console.log('No valid context found');
      curField = null;
      curTerm = '';
      return callback([]);
    },
    
    replace: function (item) {
      console.log(`Replacing with: "${item.text}" for field: "${curField}"`);
      
      const fullText = $ta.val();
      const textBeforeCaret = fullText.slice(0, $ta[0].selectionStart);
      
      // CASE 1: Multi-value context (after comma)
      const lastComma = textBeforeCaret.lastIndexOf(',');
      if (lastComma !== -1) {
        const beforeComma = fullText.slice(0, lastComma + 1);
        const afterCaret = fullText.slice($ta[0].selectionStart);
        return beforeComma + ' ' + item.text + afterCaret;
      }
      
      // CASE 2: Field context with existing term
      const fieldContext = textBeforeCaret.match(findAnyFieldContext);
      if (fieldContext && fieldContext[2].trim()) {
        const termStartPos = textBeforeCaret.lastIndexOf(fieldContext[2]);
        
        if (termStartPos !== -1) {
          const textBeforeTerm = fullText.slice(0, termStartPos);
          const textAfterTerm = fullText.slice($ta[0].selectionStart);
          
          // If we have a field context, preserve the field name
          if (curField) {
            return textBeforeTerm + item.text + ' ' + textAfterTerm;
          }
        }
      }
      
      // CASE 3: New field just started (field name but no term yet)
      const newFieldStart = textBeforeCaret.match(findNewFieldStart);
      if (newFieldStart && curField) {
        const textAfterCaret = fullText.slice($ta[0].selectionStart);
        return fullText.slice(0, $ta[0].selectionStart) + ' ' + item.text + textAfterCaret;
      }
      
      // Fallback
      return item.text + ' ';
    },
    
    template: function (item) {
      return `
        <div class="card suggest-card">
          <div class="card-body py-2 px-3">
            <div class="d-flex justify-content-between align-items-center gap-2">
              <div class="suggest-title">${highlight(item.text, curTerm)}</div>
              ${item.meta ? `<small class="suggest-meta text-truncate">${item.meta}</small>` : ``}
            </div>
          </div>
        </div>`;
    }
  }], {
    maxCount: 15,
    debounce: 300,
    zIndex: 10000,
    dropdownClassName: 'textcomplete-dropdown'
  });

  // Enable/disable Send button
  $ta.on('input', function () {
    $(SEND_BTN_SELECTOR).prop('disabled', !this.value.trim());
  });

  // Debug helper
  window.debugAutocomplete = function() {
    const text = $ta.val().slice(0, $ta[0].selectionStart);
    console.log('=== DEBUG AUTCOMPLETE ===');
    console.log('Current text:', text);
    
    const fieldContext = text.match(findAnyFieldContext);
    console.log('Field context:', fieldContext);
    
    const newFieldStart = text.match(findNewFieldStart);
    console.log('New field start:', newFieldStart);
    
    const lastComma = text.lastIndexOf(',');
    console.log('Last comma position:', lastComma);
    
    if (lastComma !== -1) {
      console.log('Text after comma:', text.slice(lastComma + 1).trim());
    }
  };
})();