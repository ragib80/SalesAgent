'use strict';
// ---------- Small helpers: loading spinner inside a button ----------
function setBtnLoading($btn, loadingText = "Generating...") {
  if (!$btn.data("original-html")) {
    $btn.data("original-html", $btn.html());
  }
  $btn.prop("disabled", true).html(
    '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>' +
    loadingText
  );
}

function norm(s) {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}
function startsWithNormalized(text, prefix) {
  if (!prefix) return false;
  return norm(text).startsWith(norm(prefix));
}
// Looks like a pure filter summary (starts with a Label:, e.g., Dealer:, Brand:, Product Name:)
function looksLikeLabelStart(s) {
  return /^[A-Za-z][A-Za-z0-9 _/-]*:\s/i.test((s || "").trimStart());
}
// Merge manual prefix with a server prompt, but only when it needs it.
function mergePrefix(prefix, txt) {
  if (!prefix) return txt;
  const t = (txt || "");
  // If the returned prompt already begins with the manual prefix (case/space insensitive) → don't add it again
  if (startsWithNormalized(t, prefix)) return t;
  // If it starts with a label block (Dealer:, Brand:, etc.) → it's a pure summary, so prepend prefix
  if (looksLikeLabelStart(t)) {
    const needsSpace = prefix.length > 0 && !/\s$/.test(prefix);
    return `${prefix}${needsSpace ? " " : ""}${t}`;
  }
  // Otherwise it's already a full sentence (e.g., "What are the sales for ...") → don't prepend
  return t;
}


function unsetBtnLoading($btn) {
  const original = $btn.data("original-html") || $btn.text() || "Generate";
  $btn.prop("disabled", false).html(original);
}

// ---------- Manual prefix cache ----------
let manualPrefixCache = "";

// If your chat input has a different selector, add it here:
const PREFIX_INPUT_SELECTORS = [
  "#message-input",
  "#chat-input",
  "#prompt-input",
  "#composer",
  "textarea[name='message']",
  "input[name='message']",
].join(", ");

// Extract "manual part" (text before first filter label) from any string
function extractManualPrefix(text) {
  const s = (text || "").toString();
  // Match common filter patterns like "Label: value" or "Label:  value"
  const firstLabelIdx = s.search(/[A-Z][A-Za-z0-9 _/-]*:\s+/);
  return firstLabelIdx >= 0 ? s.slice(0, firstLabelIdx).trim() : s.trim();
}

// Snapshot prefix source -> cache
function refreshManualPrefixCache() {
  // Prefer existing textarea's manual part (user might have edited there)
  const $ta = $("#generatedPromptsContainer textarea").first();
  if ($ta.length) {
    manualPrefixCache = extractManualPrefix($ta.val());
    return;
  }

  // Try visible chat inputs in order
  const $inputs = $(PREFIX_INPUT_SELECTORS).filter(":visible");
  for (let i = 0; i < $inputs.length; i++) {
    const val = $inputs.eq(i).val();
    if (val && val.toString().length) {
      manualPrefixCache = extractManualPrefix(val);
      return;
    }
  }
  // If nothing found, keep existing cache (last good value) instead of wiping it
}

// Always use cache unless a textarea exists (from which we can live-read)
function getManualPrefix() {
  const $ta = $("#generatedPromptsContainer textarea").first();
  if ($ta.length) return extractManualPrefix($ta.val());

  // Fallback: ensure cache is fresh before using it
  if (!manualPrefixCache) refreshManualPrefixCache();
  return manualPrefixCache || "";
}

$(function () {
  // ---------- Auth wrapper ----------
  function getAuthToken() {
    return localStorage.getItem("auth_token");
  }

  function sendAuthenticatedRequest(options) {
    const token = getAuthToken();
    if (!token) {
      Swal.fire({
        icon: "warning",
        title: "Not logged in",
        text: "Please log in to use Prompt Helper.",
      });
      return;
    }

    $.ajax({
      ...options,
      headers: { Authorization: "Bearer " + token },
      error: function (xhr, status, err) {
        if (xhr.status === 401) {
          Swal.fire({
            icon: "error",
            title: "Unauthorized",
            text: "Your session has expired. Please log in again.",
          }).then(() => {
            localStorage.clear();
            window.location.href = "/welcome";
          });
        } else if (options.error) {
          options.error(xhr, status, err);
        }
      },
    });
  }

  // ---------- Capture/refresh prefix at the right moments ----------
  const offcanvasEl = document.getElementById("promptHelperCanvas");
  if (offcanvasEl) {
    offcanvasEl.addEventListener("show.bs.offcanvas", refreshManualPrefixCache);
    offcanvasEl.addEventListener("shown.bs.offcanvas", refreshManualPrefixCache);
  }

  // Keep cache synced as user types in the chat input
  $(document).on("input", PREFIX_INPUT_SELECTORS, refreshManualPrefixCache);

  // Keep cache synced if user edits the generated textarea
  $(document).on("input", "#generatedPromptsContainer textarea", function () {
    manualPrefixCache = extractManualPrefix($(this).val());
  });

  // ---------- Init the "available columns" selector ----------
  $("#availableColumns").select2({
    placeholder: "Select columns...",
    closeOnSelect: false,
    allowClear: true,
    width: "100%",
  });

  // ---------- Live preview builder (kept top-level & bound once) ----------
  function updateGeneratedPromptPreview() {
    // Final fallback: ensure we have the latest prefix before building
    refreshManualPrefixCache();

    const filters = {};
    const selectedCols = $("#availableColumns").val() || [];

    // Build dictionary of active filters
    selectedCols.forEach((col) => {
      if (col === "fkdat") {
        const from = $("#date-from").val();
        const to = $("#date-to").val();
        if (from || to) {
          const dateRange = from && to ? `${from} to ${to}` : (from || to);
          filters[col] = [dateRange];
        }
      } else {
        const values = $(`#select-${col}`).val();
        if (values && values.length > 0) filters[col] = values;
      }
    });

    // Manual prefix from existing textarea or from cached chat input
    let baseTextRaw = getManualPrefix();

    // Human-readable summary with proper formatting
    const readableParts = [];
    for (const [col, vals] of Object.entries(filters)) {
      const label = $(`#availableColumns option[value='${col}']`).text() || col;
      readableParts.push(`${label}: ${vals.join(", ")}`);
    }
    const filterSummary = readableParts.join(", ");

    // Build combined text: "manual prefix" + " " + "filters"
    let combined = baseTextRaw;
    if (filterSummary) {
      // Add space after base text if it doesn't end with space
      if (combined && !combined.endsWith(" ")) {
        combined += " ";
      }
      combined += filterSummary;
    }

    const promptBox = $("#generatedPromptsContainer textarea").first();
    if (promptBox.length) {
      // Only update if content actually changed to avoid cursor jump
      const currentText = promptBox.val() || "";
      if (combined !== currentText) {
        promptBox.val(combined);
      }
    } else {
      // Create the box with base + filters
      $("#generatedPromptsWrapper").show();
      $("#generatedPromptsContainer").html(`
        <div class="mb-3">
          <textarea class="form-control prompt-textarea mb-2" rows="3">${combined}</textarea>
          <div class="d-flex gap-2 flex-wrap align-items-center">
            <button class="btn btn-sm btn-outline-secondary copy-btn">📋 Copy</button>
            <button class="btn btn-sm btn-outline-success use-prompt-btn">➡️ Use Prompt</button>
            <button class="btn btn-sm btn-outline-danger clear-prompt-btn">🗑️ Clear</button>
          </div>
        </div>
      `);
    }
  }

  // ---------- Render per-column filters when columns change ----------
  $("#availableColumns").on("change", function () {
    const selectedCols = $(this).val() || [];
    const container = $("#selectedFiltersContainer");

    // Remove deselected filters
    container.children(".filter-block").each(function () {
      const col = $(this).data("col");
      if (!selectedCols.includes(col)) {
        $(this).remove();
      }
    });

    // Add newly selected filters
    selectedCols.forEach((col) => {
      if (container.find(`.filter-block[data-col="${col}"]`).length === 0) {
        const label = $(`#availableColumns option[value="${col}"]`).text();

        if (col === "fkdat") {
          container.append(`
            <div class="filter-block mb-3 p-3 border rounded bg-light shadow-sm" data-col="${col}">
              <label class="form-label fw-semibold mb-2">${label} (From - To)</label>
              <div class="row g-2">
                <div class="col">
                  <input type="date" class="form-control" id="date-from" />
                </div>
                <div class="col">
                  <input type="date" class="form-control" id="date-to" />
                </div>
              </div>
            </div>
          `);
        } else {
          container.append(`
            <div class="filter-block mb-3 p-3 border rounded bg-light shadow-sm" data-col="${col}">
              <label class="form-label fw-semibold mb-2">${label}</label>
              <select id="select-${col}" class="form-select" multiple></select>
            </div>
          `);

          $(`#select-${col}`).select2({
            placeholder: `Choose ${label}...`,
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
              processResults: function (data) {
                return {
                  results: data.results || [],
                  pagination: { more: data.pagination?.more || false },
                };
              },
            },
          });
        }
      }
    });

    // After adding/removing blocks, update preview once
    updateGeneratedPromptPreview();
  });

  // ---------- Watchers (bind once globally) ----------
  $(document).on("change", ".form-select", updateGeneratedPromptPreview);
  $(document).on("select2:select select2:unselect", ".form-select", updateGeneratedPromptPreview);
  $(document).on("change", "#date-from, #date-to", updateGeneratedPromptPreview);

  // ---------- Apply / Generate ----------
  $("#applyFiltersBtn").on("click", function () {
    const $btn = $(this);
    
    // FIRST: Capture the manual prefix before we do anything else
    refreshManualPrefixCache();
    
    setBtnLoading($btn, "Generating...");

    const selectedCols = $("#availableColumns").val() || [];
    const filters = {};
    let abort = false;

    selectedCols.forEach((col) => {
      if (col === "fkdat") {
        const from = $("#date-from").val();
        const to = $("#date-to").val();

        if (!from && !to) {
          Swal.fire({
            icon: "warning",
            title: "Date filter missing",
            text: "Please select at least one date or remove the Date filter.",
          });
          abort = true;
          return;
        }

        function formatDate(d) {
          if (!d) return null;
          const date = new Date(d);
          const options = { day: "2-digit", month: "long", year: "numeric" };
          return date.toLocaleDateString("en-GB", options);
        }

        if (from && !to) {
          filters[col] = [formatDate(from)];
        } else {
          filters[col] = [`${formatDate(from)} to ${formatDate(to)}`];
        }
      } else {
        const values = $(`#select-${col}`).val();
        if (values && values.length > 0) {
          filters[col] = values;
        }
      }
    });

    if (abort) {
      unsetBtnLoading($btn);
      return;
    }

    const metric = $("#metricSelect").val();
    const generatedPrompt = $(".prompt-textarea").val();

    sendAuthenticatedRequest({
      url: "/api/sales/apply-filters/",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ filters, metric, generatedPrompt }),
      success: function (resp) {
        const container = $("#generatedPromptsContainer");
        $("#generatedPromptsWrapper").show();

        // latest saved manual prefix (e.g., "what is the sales of  ")
        const savedPrefix = manualPrefixCache || "";

        // helper to render one prompt block
        const renderBlock = (text, idx = 0) => {
          container.append(`
            <div class="mb-3">
              <textarea class="form-control prompt-textarea mb-2" rows="3" id="prompt-${idx}">${text}</textarea>
              <div class="d-flex gap-2">
                <button class="btn btn-sm btn-outline-secondary copy-btn" data-target="prompt-${idx}">📋 Copy</button>
                <button class="btn btn-sm btn-outline-success use-prompt-btn" data-target="prompt-${idx}">➡️ Use Prompt</button>
                <button class="btn btn-sm btn-outline-danger clear-prompt-btn" data-target="prompt-${idx}">🗑️ Clear</button>
              </div>
            </div>
          `);
        };

        if (Array.isArray(resp.prompts) && resp.prompts.length > 0) {
          if (resp.prompts.length > 1) {
            // MULTIPLE: clear first, then append each
            container.find("textarea.prompt-textarea").val(""); // visual clear (optional)
            container.empty();

            resp.prompts.forEach((p, idx) => {
              const finalText = mergePrefix(savedPrefix, p);
              renderBlock(finalText, idx);
            });
          } else {
            // SINGLE: update existing textarea if present; otherwise render one fresh
            const finalText = mergePrefix(savedPrefix, resp.prompts[0]);
            const existing = container.find("textarea.prompt-textarea").first();

            if (existing.length) {
              existing.val(finalText);
            } else {
              container.empty();
              renderBlock(finalText, 0);
            }
          }
        } else {
          container.empty().append(
            `<div class="text-muted">No prompt could be generated. Please adjust your filters.</div>`
          );
        }
      },
      error: function (xhr, status, err) {
        console.error(err);
      },
      complete: function () {
        unsetBtnLoading($btn);
      },
    });
  });

  // ---------- Delegated actions: Copy / Use Prompt / Clear All ----------
  // COPY
  $(document).on("click", ".copy-btn", function () {
    const targetId = $(this).data("target");
    const $ta = targetId ? $(`#${targetId}`) : $(this).closest(".mb-3").find("textarea");
    const text = ($ta.val() || "").toString();

    navigator.clipboard.writeText(text).then(() => {
      Swal.fire({
        icon: "success",
        title: "Copied!",
        text: "Prompt copied to clipboard.",
        timer: 1500,
        showConfirmButton: false,
      });
    });
  });

  // USE PROMPT
  $(document).on("click", ".use-prompt-btn", function () {
    const targetId = $(this).data("target");
    const $ta = targetId ? $(`#${targetId}`) : $(this).closest(".mb-3").find("textarea");
    const text = ($ta.val() || "").toString();

    $("#message-input").val(text);
    $("#use-prompt-btn").prop("disabled", false).trigger("click");
    $("#send-btn").prop("disabled", false).trigger("click");

    const offcanvasEl2 = document.getElementById("promptHelperCanvas");
    if (offcanvasEl2) {
      const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(offcanvasEl2);
      offcanvas.hide();
    }
  });

  // CLEAR ALL
  $(document).on("click", ".clear-prompt-btn", function () {
    Swal.fire({
      title: "Clear All?",
      text: "This will remove all selected filters, metric, and prompts.",
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "Yes, clear all",
      cancelButtonText: "Cancel",
    }).then((res) => {
      if (res.isConfirmed) {
        $("#availableColumns").val(null).trigger("change");
        $("#selectedFiltersContainer").empty();
        $("#metricSelect").val("");
        $("#generatedPromptsContainer").empty();
        $("#generatedPromptsWrapper").hide();
        manualPrefixCache = ""; // Clear the cache too

        Swal.fire({
          icon: "info",
          title: "Cleared",
          text: "All filters and prompts have been cleared.",
          timer: 1200,
          showConfirmButton: false,
        });
      }
    });
  });
});