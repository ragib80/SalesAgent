$(function () {
  // 🔑 Helper: Get token from localStorage
  function getAuthToken() {
    return localStorage.getItem("auth_token");
  }

  // 🔑 Helper: Centralized AJAX wrapper with token + error handling
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
            window.location.href = "/login";
          });
        } else if (options.error) {
          options.error(xhr, status, err);
        }
      },
    });
  }

  // Init multi-select for choosing columns
  $("#availableColumns").select2({
    placeholder: "Select columns...",
    closeOnSelect: false,
    allowClear: true,
    width: "100%",
  });

  // Render value dropdowns below
  $("#availableColumns").on("change", function () {
    const selectedCols = $(this).val() || [];
    const container = $("#selectedFiltersContainer");
    container.empty();

    selectedCols.forEach((col) => {
      const label = $(`#availableColumns option[value="${col}"]`).text();

      container.append(`
        <div class="mb-3 p-3 border rounded bg-light shadow-sm">
          <label class="form-label fw-semibold mb-2">${label}</label>
          <select id="select-${col}" class="form-select" multiple></select>
        </div>
      `);

      // Initialize Select2 with dynamic AJAX
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
    });
  });

  // Apply Filters
  $("#applyFiltersBtn").on("click", function () {
    const selectedCols = $("#availableColumns").val() || [];
    const filters = {};

    selectedCols.forEach((col) => {
      const values = $(`#select-${col}`).val();
      if (values && values.length > 0) {
        filters[col] = values; // col = SAP field, values = selected
      }
    });

    sendAuthenticatedRequest({
      url: "/api/sales/apply-filters/",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ filters }),
      success: function (resp) {
        console.log("✅ Filters applied:", resp);

        const container = $("#generatedPromptsContainer");
        container.empty();

        if (resp.prompts && resp.prompts.length > 0) {
             $("#generatedPromptsWrapper").show(); 
          resp.prompts.forEach((p, idx) => {
            container.append(`
              <div class="d-flex align-items-start mb-2">
                <textarea class="form-control prompt-textarea flex-grow-1" 
                  rows="2" id="prompt-${idx}">${p}</textarea>
                <button class="btn btn-sm btn-outline-secondary ms-2 copy-btn" data-target="prompt-${idx}">
                  📋
                </button>
              </div>
            `);
          });

          // Copy button action
          $(".copy-btn").on("click", function () {
            const targetId = $(this).data("target");
            const text = $(`#${targetId}`).val();
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
        } else {
          container.append(
            `<div class="text-muted">No prompt could be generated. Please adjust your filters.</div>`
          );
        }
      },
    });
  });
});
