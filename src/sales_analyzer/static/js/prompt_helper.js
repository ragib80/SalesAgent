$(function () {
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
            window.location.href = "/login";
          });
        } else if (options.error) {
          options.error(xhr, status, err);
        }
      },
    });
  }

  // Init column selector
  $("#availableColumns").select2({
    placeholder: "Select columns...",
    closeOnSelect: false,
    allowClear: true,
    width: "100%",
  });

  // Render value dropdowns
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

    // Add new filters
    selectedCols.forEach((col) => {
      if (container.find(`.filter-block[data-col="${col}"]`).length === 0) {
        const label = $(`#availableColumns option[value="${col}"]`).text();

        if (col === "fkdat") {
          // Date range special case
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
          // Regular Select2 dropdown
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
  });

  // Apply filters
  $("#applyFiltersBtn").on("click", function () {
    const selectedCols = $("#availableColumns").val() || [];
    const filters = {};

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
              <div class="mb-3">
                <textarea class="form-control prompt-textarea mb-2"
                  rows="2" id="prompt-${idx}">${p}</textarea>
                <div class="d-flex gap-2">
                  <button class="btn btn-sm btn-outline-secondary copy-btn" data-target="prompt-${idx}">
                    📋 Copy
                  </button>
                  <button class="btn btn-sm btn-outline-success use-prompt-btn" data-target="prompt-${idx}">
                    ➡️ Use Prompt
                  </button>
                </div>
              </div>
            `);
          });




          // Copy action
          $(".copy-btn").on("click", function () {
            const text = $(`#${$(this).data("target")}`).val();
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

          // Send to Chat action
          $(".use-prompt-btn").on("click", function () {
            const text = $(`#${$(this).data("target")}`).val();
            $("#message-input").val(text);
            $("#use-prompt-btn").prop("disabled", false).trigger("click");

              $("#send-btn").prop("disabled", false).trigger("click");

            const offcanvasEl = document.getElementById("promptHelperCanvas");
            const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(offcanvasEl);
            offcanvas.hide();
          });
        } else {
          $("#generatedPromptsWrapper").show();
          container.append(
            `<div class="text-muted">No prompt could be generated. Please adjust your filters.</div>`
          );
        }
      },
    });
  });
});
