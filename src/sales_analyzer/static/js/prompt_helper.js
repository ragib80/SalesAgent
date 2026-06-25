'use strict';

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
function norm(s) {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}
function startsWithNormalized(text, prefix) {
  return prefix ? norm(text).startsWith(norm(prefix)) : false;
}
function looksLikeLabelStart(s) {
  return /^[A-Za-z][A-Za-z0-9 _/-]*:\s/i.test((s || "").trimStart());
}
function mergePrefix(prefix, txt) {
  if (!prefix) return txt;
  const t = txt || "";
  if (startsWithNormalized(t, prefix)) return t;
  if (looksLikeLabelStart(t)) {
    return `${prefix}${/\s$/.test(prefix) ? "" : " "}${t}`;
  }
  return t;
}

function setBtnLoading($btn, text = "Generating…") {
  $btn.data("original-html", $btn.html()).prop("disabled", true).html(
    `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${text}`
  );
}
function unsetBtnLoading($btn) {
  $btn.prop("disabled", false).html($btn.data("original-html") || "Generate");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Manual-prefix cache (synced from chat input or live textarea)
// ─────────────────────────────────────────────────────────────────────────────
let manualPrefixCache = "";

const PREFIX_INPUT_SELECTORS = [
  "#message-input", "#chat-input", "#prompt-input",
  "#composer", "textarea[name='message']", "input[name='message']",
].join(", ");

function extractManualPrefix(text) {
  const s = (text || "").toString();
  const idx = s.search(/[A-Z][A-Za-z0-9 _/-]*:\s+/);
  return idx >= 0 ? s.slice(0, idx).trim() : s.trim();
}
function refreshManualPrefixCache() {
  const $inputs = $(PREFIX_INPUT_SELECTORS).filter(":visible");
  for (let i = 0; i < $inputs.length; i++) {
    const val = $inputs.eq(i).val();
    if (val && val.length) { manualPrefixCache = extractManualPrefix(val); return; }
  }
}
function getManualPrefix() {
  if (!manualPrefixCache) refreshManualPrefixCache();
  return manualPrefixCache || "";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Render a generated prompt as a card (no editable textarea)
// ─────────────────────────────────────────────────────────────────────────────
let _cardIdx = 0;
function renderPromptCard(text) {
  const id = `ph-card-${++_cardIdx}`;
  return `
    <div class="ph-prompt-card" id="${id}" data-prompt="${text.replace(/"/g, '&quot;')}">
      <p class="ph-prompt-card-text mb-0">${text}</p>
      <div class="ph-prompt-card-actions">
        <button class="ph-card-btn ph-card-use" data-card="${id}" title="Insert into chat">
          <i class="bi bi-arrow-right-circle me-1"></i>Use
        </button>
        <button class="ph-card-btn ph-card-copy" data-card="${id}" title="Copy to clipboard">
          <i class="bi bi-clipboard me-1"></i>Copy
        </button>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Insert prompt into chat input (no auto-send)
// ─────────────────────────────────────────────────────────────────────────────
function usePrompt(text) {
  const $input = $("#message-input");
  $input.val(text).trigger("input");
  // Enable send button
  $("#send-btn").prop("disabled", false);
  // Close offcanvas
  const el = document.getElementById("promptHelperCanvas");
  if (el) bootstrap.Offcanvas.getOrCreateInstance(el).hide();
  // Focus the input so user can review before hitting Enter
  $input.focus();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Auth wrapper
// ─────────────────────────────────────────────────────────────────────────────
function getAuthToken() { return localStorage.getItem("auth_token"); }

function sendAuthenticatedRequest(options) {
  const token = getAuthToken();
  if (!token) {
    Swal.fire({ icon: "warning", title: "Not logged in", text: "Please log in to use Prompt Helper." });
    return;
  }
  $.ajax({
    ...options,
    headers: { Authorization: "Bearer " + token },
    error: function (xhr, status, err) {
      if (xhr.status === 401) {
        Swal.fire({ icon: "error", title: "Unauthorized", text: "Session expired. Please log in again." })
          .then(() => { localStorage.clear(); window.location.href = "/welcome"; });
      } else if (options.error) {
        options.error(xhr, status, err);
      }
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Filter-preview builder (live update while user picks filter values)
// ─────────────────────────────────────────────────────────────────────────────
function buildFilterSummary() {
  const parts = [];
  const selectedCols = $("#availableColumns").val() || [];
  selectedCols.forEach((col) => {
    if (col === "fkdat") {
      const from = $("#date-from").val(), to = $("#date-to").val();
      if (from || to) parts.push(`Date: ${from && to ? from + " to " + to : from || to}`);
    } else {
      const vals = $(`#select-${col}`).val();
      if (vals && vals.length) {
        const label = $(`#availableColumns option[value='${col}']`).text() || col;
        parts.push(`${label}: ${vals.join(", ")}`);
      }
    }
  });
  return parts.join(", ");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main DOM-ready
// ─────────────────────────────────────────────────────────────────────────────
$(function () {

  // ── Tab switching ──────────────────────────────────────────────────────────
  $(document).on("click", ".ph-tab-btn", function () {
    const tab = $(this).data("tab");
    $(".ph-tab-btn").removeClass("active");
    $(this).addClass("active");
    $(".ph-tab-panel").hide();
    $(`#ph-tab-${tab}`).show();
  });

  // ── Capture prefix when offcanvas opens ───────────────────────────────────
  const offcanvasEl = document.getElementById("promptHelperCanvas");
  if (offcanvasEl) {
    offcanvasEl.addEventListener("show.bs.offcanvas", refreshManualPrefixCache);
    offcanvasEl.addEventListener("shown.bs.offcanvas", refreshManualPrefixCache);
  }
  $(document).on("input", PREFIX_INPUT_SELECTORS, refreshManualPrefixCache);

  // ── Init the column multi-select ──────────────────────────────────────────
  $("#availableColumns").select2({
    placeholder: "Choose filter columns…",
    closeOnSelect: false,
    allowClear: true,
    width: "100%",
  });

  // ── Quick Template chips ───────────────────────────────────────────────────
  $(document).on("click", ".ph-chip", function () {
    const $chip = $(this);
    const metric = $chip.data("tpl-metric");
    const col    = $chip.data("tpl-col");
    const text   = $chip.data("tpl-text");

    // Highlight active chip
    $(".ph-chip").removeClass("active");
    $chip.addClass("active");

    // Pre-fill metric
    if (metric) $("#metricSelect").val(metric);

    // Pre-select column (if any)
    if (col) {
      const current = $("#availableColumns").val() || [];
      if (!current.includes(col)) {
        current.push(col);
        $("#availableColumns").val(current).trigger("change");
      }
    }

    // Populate and show the preview immediately
    _showPreviewCard(text);
  });

  // ── Column change → render filter blocks ──────────────────────────────────
  $("#availableColumns").on("change", function () {
    const selectedCols = $(this).val() || [];
    const container = $("#selectedFiltersContainer");

    // Remove deselected
    container.children(".filter-block").each(function () {
      if (!selectedCols.includes($(this).data("col"))) $(this).remove();
    });

    // Add new
    selectedCols.forEach((col) => {
      if (container.find(`.filter-block[data-col="${col}"]`).length) return;
      const label = $(`#availableColumns option[value="${col}"]`).text();

      if (col === "fkdat") {
        container.append(`
          <div class="filter-block mb-3 p-3 border rounded" data-col="${col}">
            <label class="form-label fw-semibold small mb-2">${label} (From – To)</label>
            <div class="row g-2">
              <div class="col"><input type="date" class="form-control form-control-sm" id="date-from"></div>
              <div class="col"><input type="date" class="form-control form-control-sm" id="date-to"></div>
            </div>
          </div>`);
      } else {
        container.append(`
          <div class="filter-block mb-3 p-3 border rounded" data-col="${col}">
            <label class="form-label fw-semibold small mb-2">${label}</label>
            <select id="select-${col}" class="form-select form-select-sm" multiple></select>
          </div>`);

        $(`#select-${col}`).select2({
          placeholder: `Choose ${label}…`,
          allowClear: true,
          width: "100%",
          ajax: {
            transport: function (params, success, failure) {
              sendAuthenticatedRequest({
                url: `/api/sales/filters/${col}/`,
                method: "GET",
                dataType: "json",
                data: params.data,
                success: success,
                error: failure,
              });
            },
            delay: 250,
            processResults: (data) => ({
              results: data.results || [],
              pagination: { more: data.pagination?.more || false },
            }),
          },
        });
      }
    });
  });

  // ── Generate Prompt button ─────────────────────────────────────────────────
  $("#applyFiltersBtn").on("click", function () {
    const $btn = $(this);
    refreshManualPrefixCache();

    const selectedCols = $("#availableColumns").val() || [];
    const filters = {};
    let abort = false;

    selectedCols.forEach((col) => {
      if (col === "fkdat") {
        const from = $("#date-from").val(), to = $("#date-to").val();
        if (!from && !to) {
          Swal.fire({ icon: "warning", title: "Date filter missing", text: "Select at least one date or remove the Date filter." });
          abort = true; return;
        }
        const fmt = (d) => {
          if (!d) return null;
          return new Date(d).toLocaleDateString("en-GB", { day: "2-digit", month: "long", year: "numeric" });
        };
        filters[col] = from && to ? [`${fmt(from)} to ${fmt(to)}`] : [fmt(from) || fmt(to)];
      } else {
        const vals = $(`#select-${col}`).val();
        if (vals && vals.length) filters[col] = vals;
      }
    });

    if (abort) return;

    const metric = $("#metricSelect").val();
    if (!metric && !Object.keys(filters).length) {
      Swal.fire({ icon: "info", title: "Nothing selected", text: "Choose a metric or at least one filter to generate a prompt." });
      return;
    }

    setBtnLoading($btn, "Generating…");

    sendAuthenticatedRequest({
      url: "/api/sales/apply-filters/",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ filters, metric, generatedPrompt: buildFilterSummary() }),
      success: function (resp) {
        const savedPrefix = manualPrefixCache || "";
        const container = $("#generatedPromptsContainer").empty();
        $("#generatedPromptsWrapper").show();

        const prompts = Array.isArray(resp.prompts) && resp.prompts.length ? resp.prompts : [];
        if (!prompts.length) {
          container.html('<p class="text-muted small">No prompts generated. Adjust your filters and try again.</p>');
          return;
        }
        prompts.forEach((p) => container.append(renderPromptCard(mergePrefix(savedPrefix, p))));
      },
      error: function (xhr) {
        Swal.fire({ icon: "error", title: "Error", text: xhr.responseJSON?.message || "Failed to generate prompts." });
      },
      complete: function () { unsetBtnLoading($btn); },
    });
  });

  // ── Helper: show a single preview card immediately (used by templates) ─────
  function _showPreviewCard(text) {
    const container = $("#generatedPromptsContainer").empty();
    $("#generatedPromptsWrapper").show();
    container.append(renderPromptCard(text));
  }

  // ── Clear all ──────────────────────────────────────────────────────────────
  $(document).on("click", "#clearAllPromptsBtn", function () {
    Swal.fire({
      title: "Clear all prompts?",
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "Clear",
      cancelButtonText: "Cancel",
    }).then((r) => {
      if (!r.isConfirmed) return;
      $("#availableColumns").val(null).trigger("change");
      $("#selectedFiltersContainer").empty();
      $("#metricSelect").val("");
      $("#generatedPromptsContainer").empty();
      $("#generatedPromptsWrapper").hide();
      $(".ph-chip").removeClass("active");
      manualPrefixCache = "";
    });
  });

  // ── Card: Use ──────────────────────────────────────────────────────────────
  $(document).on("click", ".ph-card-use", function () {
    const cardId = $(this).data("card");
    const text = $(`#${cardId}`).data("prompt") || $(`#${cardId}`).find(".ph-prompt-card-text").text().trim();
    usePrompt(text);
  });

  // ── Card: Copy ─────────────────────────────────────────────────────────────
  $(document).on("click", ".ph-card-copy", function () {
    const cardId = $(this).data("card");
    const text = $(`#${cardId}`).data("prompt") || $(`#${cardId}`).find(".ph-prompt-card-text").text().trim();
    navigator.clipboard.writeText(text).then(() =>
      Swal.fire({ icon: "success", title: "Copied!", timer: 1200, showConfirmButton: false })
    );
  });

  // ─────────────────────────────────────────────────────────────────────────
  //  SUGGEST TAB — AI prompt suggestions
  // ─────────────────────────────────────────────────────────────────────────
  $(document).on("click", "#ph-suggest-btn", function () {
    const $btn = $(this);
    const input = $("#ph-suggest-input").val().trim();
    if (!input) {
      Swal.fire({ icon: "info", title: "Nothing to suggest", text: "Type a partial question first." });
      return;
    }

    setBtnLoading($btn, "Thinking…");
    const conversationId = $("#chat-id-holder").data("current-conversation-id") || null;

    sendAuthenticatedRequest({
      url: "/api/sales/prompt-suggestions/",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ input_text: input, conversation_id: conversationId }),
      success: function (resp) {
        const results = $("#ph-suggest-results").empty();
        const suggestions = resp.suggestions || [];
        if (!suggestions.length) {
          results.html('<p class="text-muted small">No suggestions returned. Try a different phrase.</p>');
          return;
        }
        suggestions.forEach((s) => {
          results.append(`
            <div class="ph-suggest-item" data-text="${s.replace(/"/g, '&quot;')}">
              <span class="ph-suggest-item-text">${s}</span>
              <button class="ph-suggest-use-btn" title="Use this prompt">
                <i class="bi bi-arrow-right-circle"></i>
              </button>
            </div>`);
        });
      },
      error: function () {
        $("#ph-suggest-results").html('<p class="text-danger small">Failed to fetch suggestions.</p>');
      },
      complete: function () { unsetBtnLoading($btn); },
    });
  });

  // Allow Enter (without Shift) in the suggest textarea to trigger suggest
  $(document).on("keydown", "#ph-suggest-input", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#ph-suggest-btn").trigger("click");
    }
  });

  // Use a suggestion
  $(document).on("click", ".ph-suggest-use-btn", function () {
    const text = $(this).closest(".ph-suggest-item").data("text");
    usePrompt(text);
  });

  // Click anywhere on the suggest item (not just the button)
  $(document).on("click", ".ph-suggest-item-text", function () {
    const text = $(this).closest(".ph-suggest-item").data("text");
    usePrompt(text);
  });
});
