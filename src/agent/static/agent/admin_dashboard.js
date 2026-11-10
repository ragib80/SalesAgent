/* Voice of Sales — admin usage dashboard. Data: agent/dashboard.py */
(function () {
  "use strict";

  var root = document.getElementById("vos-dash");
  if (!root || !window.echarts) return;

  var STATS_URL = root.dataset.statsUrl;
  var AUDIT_URL = root.dataset.auditUrl;
  var $ = function (id) { return document.getElementById(id); };

  // Python weekday (Mon=0) → rows ordered Saturday-first (Bangladesh work week).
  var WEEK = [[5, "Sat"], [6, "Sun"], [0, "Mon"], [1, "Tue"], [2, "Wed"], [3, "Thu"], [4, "Fri"]];
  var HOURS = Array.from({ length: 24 }, function (_, h) { return (h < 10 ? "0" : "") + h + ":00"; });

  var OUTCOME_STYLE = {
    answered:       { icon: "fa-check-circle",       color: "good",     q: { outcome: "answered" } },
    general:        { icon: "fa-comment-dots",       color: "s1",       q: { outcome: "general" } },
    no_data:        { icon: "fa-inbox",              color: "warning",  q: { outcome: "no_data" } },
    kql_failed:     { icon: "fa-code",               color: "critical", q: { outcome: "kql_failed" } },
    adx_failed:     { icon: "fa-database",           color: "critical", q: { outcome: "adx_failed" } },
    summary_failed: { icon: "fa-file-alt",           color: "serious",  q: { outcome: "summary_failed" } },
    access_denied:  { icon: "fa-lock",               color: "serious",  q: { outcome: "access_denied" } },
    error:          { icon: "fa-bug",                color: "critical", q: { outcome: "error" } }
  };
  var ALWAYS_SHOWN = ["answered", "general", "no_data", "kql_failed", "adx_failed"];
  var VALIDATION_STYLE = {
    valid:        { icon: "fa-check",       color: "good",     label: "Valid on first try" },
    repaired:     { icon: "fa-wrench",      color: "s1",       label: "Repaired before ADX" },
    adx_repaired: { icon: "fa-tools",       color: "warning",  label: "Repaired after ADX error" },
    failed:       { icon: "fa-times",       color: "critical", label: "Could not produce valid KQL" }
  };

  var state = readState();
  var data = null;
  var charts = {};
  var latestTab = "latest";
  var inflight = null;

  // ── Helpers ────────────────────────────────────────────────────────────────
  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue("--vos-" + name).trim();
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  var nf = new Intl.NumberFormat("en-US");
  var nfCompact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
  function num(v) { return v == null ? "–" : nf.format(v); }
  function compact(v) { return v == null ? "–" : (v >= 10000 ? nfCompact.format(v) : nf.format(v)); }
  function secs(ms) {
    if (ms == null) return "–";
    return ms >= 1000 ? (ms / 1000).toFixed(ms >= 10000 ? 0 : 1) + " s" : Math.round(ms) + " ms";
  }
  function pct(part, whole) { return whole ? Math.round(part / whole * 1000) / 10 : 0; }
  function usd(v) {
    if (v == null) return "–";
    if (v === 0) return "$0.00";
    return "$" + (v >= 100 ? nf.format(Math.round(v)) : v.toFixed(v >= 1 ? 2 : 3));
  }
  function fmtDate(iso) {
    var d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  }
  var rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  function ago(iso) {
    var s = (new Date(iso) - Date.now()) / 1000;
    var units = [["year", 31536000], ["month", 2592000], ["day", 86400], ["hour", 3600], ["minute", 60]];
    for (var i = 0; i < units.length; i++) {
      if (Math.abs(s) >= units[i][1]) return rtf.format(Math.round(s / units[i][1]), units[i][0]);
    }
    return "just now";
  }
  // Today's date in Bangladesh time (the dashboard's day boundaries), as YYYY-MM-DD.
  function dhakaToday() {
    var p = {};
    new Intl.DateTimeFormat("en-US", { timeZone: "Asia/Dhaka", year: "numeric", month: "2-digit", day: "2-digit" })
      .formatToParts(new Date())
      .forEach(function (x) { p[x.type] = x.value; });
    return p.year + "-" + p.month + "-" + p.day;
  }
  function fullTime(iso) {
    return new Date(iso).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  }

  /**
   * Delta chip. `better` = "up" | "down" | "none" decides the good/bad tint;
   * the arrow always shows the raw direction so color is never the only cue.
   */
  function delta(value, better, unit) {
    if (value == null) return '<span class="vos-delta is-flat" title="No data for the previous period">new</span>';
    var dir = value > 0 ? "up" : value < 0 ? "down" : "flat";
    var cls = dir === "flat" || better === "none" ? "is-flat" : (dir === better ? "is-good" : "is-bad");
    var icon = dir === "up" ? "fa-arrow-up" : dir === "down" ? "fa-arrow-down" : "fa-minus";
    var txt = Math.abs(value) + (unit || "%");
    return '<span class="vos-delta ' + cls + '" title="vs previous period"><i class="fas ' + icon + '" aria-hidden="true"></i>' + txt + "</span>";
  }

  function bucketHeading() {
    return { hour: "Hour", day: "Date", week: "Week starting" }[data.range.bucket];
  }

  function auditLink(extra) {
    var p = new URLSearchParams();
    if (data) {
      // Exact Bangladesh-time bounds from the server (end exclusive), e.g. 2026-11-03T00:00:00+06:00
      p.set("created_at__gte", data.range.start_at);
      p.set("created_at__lt", data.range.end_at);
    }
    if (state.user) p.set("user", state.user);
    Object.keys(extra || {}).forEach(function (k) { p.set(k, extra[k]); });
    return AUDIT_URL + "?" + p.toString();
  }

  // ── State <-> URL ──────────────────────────────────────────────────────────
  function readState() {
    var p = new URLSearchParams(location.search);
    return {
      range: p.get("range") || "30d",
      start: p.get("start") || "",
      end: p.get("end") || "",
      user: p.get("user") || ""
    };
  }
  function query() {
    var p = new URLSearchParams({ range: state.range });
    if (state.range === "custom") { p.set("start", state.start); p.set("end", state.end); }
    if (state.user) p.set("user", state.user);
    return p.toString();
  }

  function syncControls() {
    document.querySelectorAll("#vos-range button").forEach(function (b) {
      var on = b.dataset.range === state.range;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", on);
    });
    $("vos-custom").hidden = state.range !== "custom";
    $("vos-user").value = state.user;
  }

  // ── Fetch ──────────────────────────────────────────────────────────────────
  function load() {
    if (inflight) inflight.abort();
    inflight = new AbortController();
    root.classList.add("is-loading");
    $("vos-refresh").classList.add("is-loading");
    history.replaceState(null, "", "?" + query());

    fetch(STATS_URL + "?" + query(), {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal: inflight.signal
    })
      .then(function (r) {
        if (!r.ok) {
          return r.json().catch(function () { return {}; }).then(function (body) {
            throw new Error(body.detail || "Server responded with " + r.status);
          });
        }
        return r.json();
      })
      .then(function (json) {
        data = json;
        $("vos-error").hidden = true;
        render();
      })
      .catch(function (err) {
        if (err.name === "AbortError") return;
        var box = $("vos-error");
        box.textContent = "Couldn't load dashboard data: " + err.message;
        box.hidden = false;
      })
      .finally(function () {
        root.classList.remove("is-loading");
        $("vos-refresh").classList.remove("is-loading");
      });
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    renderHeader();
    renderUserFilter();
    renderKpis();
    renderActivity();
    renderOutcomes();
    renderUsers();
    renderShare();
    renderLatency();
    renderHeatmap();
    renderLatest();
    renderQuality();
    refreshOpenTables();
  }

  function renderHeader() {
    var r = data.range;
    var label = r.start === r.end ? fmtDate(r.start) : fmtDate(r.start) + " – " + fmtDate(r.end);
    $("vos-period-label").textContent = label + " · compared with the previous period of equal length";
    $("vos-updated").textContent = "Updated " + new Date(r.generated_at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    var per = { hour: "Hourly", day: "Daily", week: "Weekly" }[r.bucket];
    $("activity-sub").textContent = per + " total, successful and failed queries";
    $("latency-sub").textContent = per + " average seconds per pipeline stage";
  }

  function renderUserFilter() {
    var sel = $("vos-user");
    var current = state.user;
    var opts = ['<option value="">All users</option>'];
    data.filters.users.forEach(function (u) {
      opts.push('<option value="' + u.id + '">' + esc(u.name) + "</option>");
    });
    sel.innerHTML = opts.join("");
    sel.value = current;
  }

  function renderKpis() {
    var k = data.kpis;
    $("kpi-total").textContent = num(k.total.value);
    $("kpi-total-foot").innerHTML = delta(k.total.delta, "up") + "<span>vs previous period</span>";

    $("kpi-success").textContent = num(k.success.value);
    $("kpi-success-foot").innerHTML = delta(k.success.delta_pts, "up", " pts") +
      "<span>" + (k.success.rate == null ? "–" : k.success.rate + "%") + " success rate</span>";

    $("kpi-failed").textContent = num(k.failed.value);
    $("kpi-failed-foot").innerHTML = delta(k.failed.delta, "down") +
      "<span>" + (k.failed.rate == null ? "–" : k.failed.rate + "%") + " of queries</span>";

    $("kpi-users").innerHTML = num(k.users.value) + "<small>/ " + num(k.users.registered) + "</small>";
    $("kpi-users-foot").innerHTML = delta(k.users.delta, "up") + "<span>of registered users</span>";

    $("kpi-latency").textContent = secs(k.latency.avg_ms);
    $("kpi-latency-foot").innerHTML = delta(k.latency.delta, "down") + "<span>p95 " + secs(k.latency.p95_ms) + "</span>";

    $("kpi-cost").textContent = usd(k.cost.usd);
    $("kpi-cost-foot").innerHTML = delta(k.cost.delta, "none") + "<span>" + compact(k.cost.tokens) + " summary tokens</span>";
  }

  // shared ECharts chrome
  function base() {
    var ink2 = cssVar("ink-2"), muted = cssVar("muted");
    return {
      textStyle: { fontFamily: getComputedStyle(root).fontFamily, color: ink2 },
      animationDuration: 500,
      tooltip: {
        backgroundColor: cssVar("surface"),
        borderColor: cssVar("border"),
        borderWidth: 1,
        textStyle: { color: cssVar("ink"), fontSize: 12 },
        extraCssText: "box-shadow:0 6px 20px rgba(16,24,40,.12);border-radius:10px;"
      },
      legend: {
        top: 0, right: 0, icon: "circle", itemWidth: 9, itemHeight: 9, itemGap: 16,
        textStyle: { color: ink2, fontSize: 12 }
      },
      _axis: {
        axisLine: { lineStyle: { color: cssVar("axis") } },
        axisTick: { show: false },
        axisLabel: { color: muted, fontSize: 11 },
        splitLine: { lineStyle: { color: cssVar("grid") } }
      }
    };
  }
  function chart(id) {
    if (!charts[id]) charts[id] = echarts.init($(id), null, { renderer: "canvas" });
    return charts[id];
  }
  function axis(extra) {
    return Object.assign({}, base()._axis, extra);
  }
  function emptyGraphic(show, text) {
    return show ? [{
      type: "text", left: "center", top: "middle",
      style: { text: text || "No queries in this period", fill: cssVar("muted"), fontSize: 13 }
    }] : [];
  }

  function renderActivity() {
    var t = data.trend, b = base();
    var s1 = cssVar("s1"), good = cssVar("good"), bad = cssVar("critical");
    var many = t.labels.length > 45;
    chart("chart-activity").setOption({
      textStyle: b.textStyle, animationDuration: b.animationDuration,
      color: [s1, good, bad],
      legend: Object.assign({}, b.legend, { data: ["Total", "Successful", "Failed"] }),
      tooltip: Object.assign({}, b.tooltip, { trigger: "axis", axisPointer: { type: "line", lineStyle: { color: cssVar("axis") } } }),
      grid: { left: 8, right: 12, top: 36, bottom: many ? 36 : 8, containLabel: true },
      xAxis: axis({ type: "category", boundaryGap: false, data: t.labels, splitLine: { show: false } }),
      yAxis: axis({ type: "value", minInterval: 1, axisLine: { show: false } }),
      dataZoom: many ? [{ type: "inside" }, { type: "slider", height: 16, bottom: 4, borderColor: "transparent", showDetail: false }] : [],
      graphic: emptyGraphic(data.kpis.total.value === 0),
      series: [
        line("Total", t.total, s1, true),
        line("Successful", t.success, good, false),
        line("Failed", t.failed, bad, false)
      ]
    }, true);

    function line(name, values, color, area) {
      return {
        name: name, type: "line", smooth: 0.35, data: values, showSymbol: false, symbolSize: 8,
        lineStyle: { width: 2, color: color },
        itemStyle: { color: color, borderColor: cssVar("surface"), borderWidth: 2 },
        emphasis: { focus: "series" },
        areaStyle: area ? {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: hexA(color, 0.22) }, { offset: 1, color: hexA(color, 0.02) }
          ])
        } : undefined
      };
    }
  }

  function hexA(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.replace(/(.)/g, "$1$1");
    var n = parseInt(h, 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
  }

  function pipelineItem(icon, color, label, count, total, href) {
    var p = pct(count, total);
    var inner =
      '<i class="ico fas ' + icon + '" style="color:' + color + '" aria-hidden="true"></i>' +
      '<span class="lbl">' + (href ? '<a class="vos-link" style="color:inherit;font-weight:400" href="' + href + '">' + esc(label) + "</a>" : esc(label)) + "</span>" +
      '<span class="cnt">' + num(count) + "</span>" +
      '<span class="pct">' + p + "%</span>" +
      '<span class="bar" role="presentation"><span style="width:' + p + "%;--c:" + color + '"></span></span>';
    return '<li class="' + (count ? "" : "is-zero") + '">' + inner + "</li>";
  }

  function renderOutcomes() {
    var total = data.kpis.total.value;
    var html = data.outcomes
      .filter(function (o) { return o.count || ALWAYS_SHOWN.indexOf(o.key) !== -1; })
      .map(function (o) {
        var st = OUTCOME_STYLE[o.key];
        return pipelineItem(st.icon, cssVar(st.color), o.label, o.count, total, auditLink(st.q));
      });
    $("list-outcomes").innerHTML = html.join("");
    $("outcomes-sub").textContent = num(total) + " queries, by final outcome";
    $("outcomes-link").href = auditLink({ success__exact: 0 });
  }

  function renderQuality() {
    var total = data.validation.reduce(function (s, v) { return s + v.count; }, 0);
    $("list-validation").innerHTML = data.validation.map(function (v) {
      var st = VALIDATION_STYLE[v.key] || { icon: "fa-circle", color: "other", label: v.label };
      return pipelineItem(st.icon, cssVar(st.color), st.label, v.count, total,
        auditLink({ kql_validation_status__exact: v.key }));
    }).join("");
    $("mix-sales").textContent = num(data.kpis.sales_queries);
    $("mix-general").textContent = num(data.kpis.general_queries);
    $("mix-conversations").textContent = num(data.kpis.conversations);
  }

  function renderUsers() {
    var b = base();
    var top = data.users.slice(0, 10).reverse();
    var good = cssVar("good"), bad = cssVar("critical"), surface = cssVar("surface");
    var names = top.map(function (u) { return u.name; });

    // Round only the outer end of each stacked bar.
    function seg(u, key, isEnd) {
      return { value: u[key], itemStyle: { borderRadius: isEnd ? [0, 4, 4, 0] : 0 } };
    }
    chart("chart-users").setOption({
      textStyle: b.textStyle, animationDuration: b.animationDuration,
      legend: Object.assign({}, b.legend, { data: ["Successful", "Failed"] }),
      tooltip: Object.assign({}, b.tooltip, {
        trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: hexA(cssVar("axis"), 0.15) } },
        formatter: function (ps) {
          var u = top[ps[0].dataIndex];
          return "<b>" + esc(u.name) + "</b><br>" +
            marker(good) + "Successful: <b>" + num(u.success) + "</b><br>" +
            marker(bad) + "Failed: <b>" + num(u.failed) + "</b><br>" +
            "Success rate: <b>" + u.success_rate + "%</b><br>" +
            "Avg response: <b>" + secs(u.avg_latency_ms) + "</b>";
        }
      }),
      grid: { left: 8, right: 40, top: 32, bottom: 8, containLabel: true },
      xAxis: axis({ type: "value", minInterval: 1, axisLine: { show: false } }),
      yAxis: axis({
        type: "category", data: names, splitLine: { show: false },
        axisLabel: { color: cssVar("ink-2"), fontSize: 12, width: 130, overflow: "truncate" }
      }),
      graphic: emptyGraphic(!top.length),
      series: [
        {
          name: "Successful", type: "bar", stack: "q", barMaxWidth: 18, color: good,
          itemStyle: { borderColor: surface, borderWidth: 1 },
          data: top.map(function (u) { return seg(u, "success", u.failed === 0); })
        },
        {
          name: "Failed", type: "bar", stack: "q", barMaxWidth: 18, color: bad,
          itemStyle: { borderColor: surface, borderWidth: 1 },
          label: {
            show: true, position: "right", color: cssVar("ink-2"), fontSize: 11,
            formatter: function (p) { return num(top[p.dataIndex].total); }
          },
          data: top.map(function (u) { return seg(u, "failed", true); })
        }
      ]
    }, true);
  }

  function marker(color) {
    return '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:' + color + '"></span>';
  }

  function renderShare() {
    var b = base();
    var palette = ["s1", "s2", "s3", "s4", "s5"].map(cssVar);
    var users = data.users;
    var top = users.slice(0, 5).map(function (u, i) {
      return { name: u.name, value: u.total, itemStyle: { color: palette[i] } };
    });
    var rest = users.slice(5).reduce(function (s, u) { return s + u.total; }, 0);
    if (rest) top.push({ name: "Others (" + (users.length - 5) + ")", value: rest, itemStyle: { color: cssVar("other") } });
    var total = data.kpis.total.value;

    chart("chart-share").setOption({
      textStyle: b.textStyle, animationDuration: b.animationDuration,
      tooltip: Object.assign({}, b.tooltip, {
        trigger: "item",
        formatter: function (p) { return marker(p.color) + esc(p.name) + ": <b>" + num(p.value) + "</b> (" + p.percent + "%)"; }
      }),
      legend: Object.assign({}, b.legend, {
        top: "auto", bottom: 0, left: "center", right: "auto", type: "scroll",
        formatter: function (n) { return n.length > 22 ? n.slice(0, 21) + "…" : n; }
      }),
      graphic: total ? [
        { type: "text", left: "center", top: "39%", style: { text: compact(total), fill: cssVar("ink"), fontSize: 26, fontWeight: 700, textAlign: "center" } },
        { type: "text", left: "center", top: "51%", style: { text: "queries", fill: cssVar("muted"), fontSize: 12, textAlign: "center" } }
      ] : emptyGraphic(true),
      series: [{
        type: "pie", radius: ["56%", "76%"], center: ["50%", "46%"], avoidLabelOverlap: true,
        padAngle: top.length > 1 ? 1 : 0,
        itemStyle: { borderColor: cssVar("surface"), borderWidth: 2, borderRadius: 4 },
        label: { show: false }, labelLine: { show: false },
        emphasis: { scale: true, scaleSize: 5 },
        data: top
      }]
    }, true);
  }

  function renderLatency() {
    var l = data.latency, b = base();
    var cols = [cssVar("s1"), cssVar("s2"), cssVar("s3")];
    var toS = function (arr) { return arr.map(function (v) { return v == null ? "-" : Math.round(v / 100) / 10; }); };
    var names = ["KQL generation", "ADX execution", "LLM summary"];
    var hasAny = l.avg.kql != null || l.avg.adx != null || l.avg.llm != null;
    var many = l.labels.length > 45;

    chart("chart-latency").setOption({
      textStyle: b.textStyle, animationDuration: b.animationDuration,
      color: cols,
      legend: Object.assign({}, b.legend, { data: names }),
      tooltip: Object.assign({}, b.tooltip, {
        trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: hexA(cssVar("axis"), 0.15) } },
        formatter: function (ps) {
          var sum = 0, rows = ps.map(function (p) {
            var v = typeof p.value === "number" ? p.value : 0;
            sum += v;
            return marker(p.color) + p.seriesName + ": <b>" + (typeof p.value === "number" ? p.value + " s" : "–") + "</b>";
          });
          return "<b>" + ps[0].axisValue + "</b><br>" + rows.join("<br>") + "<br>Total: <b>" + (Math.round(sum * 10) / 10) + " s</b>";
        }
      }),
      grid: { left: 8, right: 12, top: 36, bottom: many ? 36 : 8, containLabel: true },
      xAxis: axis({ type: "category", data: l.labels, splitLine: { show: false } }),
      yAxis: axis({ type: "value", axisLine: { show: false }, axisLabel: { color: cssVar("muted"), fontSize: 11, formatter: "{value} s" } }),
      dataZoom: many ? [{ type: "inside" }, { type: "slider", height: 16, bottom: 4, borderColor: "transparent", showDetail: false }] : [],
      graphic: emptyGraphic(!hasAny, "No timing data in this period"),
      series: [toS(l.kql), toS(l.adx), toS(l.llm)].map(function (values, i) {
        return {
          name: names[i], type: "bar", stack: "lat", barMaxWidth: 22, data: values,
          itemStyle: { borderColor: cssVar("surface"), borderWidth: 1, borderRadius: i === 2 ? [4, 4, 0, 0] : 0 }
        };
      })
    }, true);
  }

  function renderHeatmap() {
    var b = base();
    var lookup = {};
    var max = 0;
    data.heatmap.forEach(function (c) { lookup[c[0] + ":" + c[1]] = c[2]; if (c[2] > max) max = c[2]; });
    var cells = [];
    WEEK.forEach(function (w, row) {
      for (var h = 0; h < 24; h++) {
        var v = lookup[w[0] + ":" + h];
        cells.push([h, row, v || "-"]);
      }
    });

    chart("chart-heatmap").setOption({
      textStyle: b.textStyle, animationDuration: b.animationDuration,
      tooltip: Object.assign({}, b.tooltip, {
        trigger: "item",
        formatter: function (p) {
          var v = typeof p.value[2] === "number" ? p.value[2] : 0;
          return "<b>" + WEEK[p.value[1]][1] + " " + HOURS[p.value[0]] + "</b><br>" + num(v) + " queries";
        }
      }),
      grid: { left: 8, right: 8, top: 4, bottom: 44, containLabel: true },
      xAxis: axis({ type: "category", data: HOURS.map(function (h) { return h.slice(0, 2); }), splitLine: { show: false }, axisLine: { show: false }, splitArea: { show: false } }),
      yAxis: axis({ type: "category", data: WEEK.map(function (w) { return w[1]; }), inverse: true, splitLine: { show: false }, axisLine: { show: false } }),
      visualMap: {
        min: 0, max: Math.max(max, 1), calculable: false, orient: "horizontal", left: "center", bottom: 0,
        itemWidth: 10, itemHeight: 140, text: ["More", "Less"], textGap: 8,
        textStyle: { color: cssVar("muted"), fontSize: 11 },
        inRange: { color: ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"] }
      },
      graphic: emptyGraphic(!max),
      series: [{
        type: "heatmap", data: cells,
        itemStyle: { borderColor: cssVar("surface"), borderWidth: 2, borderRadius: 3 },
        emphasis: { itemStyle: { borderColor: cssVar("ink"), borderWidth: 1 } }
      }]
    }, true);
  }

  function renderLatest() {
    var items = data[latestTab];
    var list = $("list-latest");
    if (!items.length) {
      list.innerHTML = '<li class="vos-empty"><i class="far fa-folder-open" aria-hidden="true"></i>' +
        (latestTab === "latest_failed" ? "No failed queries in this period" : "No queries in this period") + "</li>";
      return;
    }
    var labels = {};
    data.outcomes.forEach(function (o) { labels[o.key] = o.label; });
    list.innerHTML = items.map(function (q) {
      var ok = q.success;
      var meta = esc(q.user) + " · " + secs(q.latency_ms) + (q.rows != null ? " · " + num(q.rows) + " rows" : "");
      return '<li><a href="' + q.url + '" title="' + esc(q.prompt) + '">' +
        '<span class="st ' + (ok ? "ok" : "bad") + '"><i class="fas ' + (ok ? "fa-check" : "fa-times") + '" aria-hidden="true"></i></span>' +
        '<span style="min-width:0"><div class="q">' + esc(q.prompt) + '</div><div class="meta">' + meta + "</div></span>" +
        '<span class="right"><span class="vos-badge ' + (ok ? "ok" : "bad") + '">' + esc(ok ? (q.outcome === "general" ? "General" : "Success") : labels[q.outcome] || "Failed") + "</span>" +
        '<time datetime="' + q.created_at + '" title="' + fullTime(q.created_at) + '">' + ago(q.created_at) + "</time></span>" +
        "</a></li>";
    }).join("");
  }

  // ── Table view (accessible alternative for each chart) ─────────────────────
  var TABLES = {
    activity: function () {
      var t = data.trend;
      return {
        cols: [[bucketHeading()], ["Total", 1], ["Successful", 1], ["Failed", 1], ["Success rate", 1]],
        rows: t.labels.map(function (l, i) {
          return [l, num(t.total[i]), num(t.success[i]), num(t.failed[i]), t.total[i] ? pct(t.success[i], t.total[i]) + "%" : "–"];
        }).reverse()
      };
    },
    users: function () {
      return {
        cols: [["User"], ["Queries", 1], ["Successful", 1], ["Failed", 1], ["Success rate", 1], ["Avg response", 1], ["Last active", 1]],
        rows: data.users.map(function (u) {
          return [esc(u.name), num(u.total), num(u.success), num(u.failed), u.success_rate + "%", secs(u.avg_latency_ms),
            u.last_active ? '<span title="' + fullTime(u.last_active) + '">' + ago(u.last_active) + "</span>" : "–"];
        })
      };
    },
    share: function () {
      var total = data.kpis.total.value;
      return {
        cols: [["User"], ["Queries", 1], ["Share", 1]],
        rows: data.users.map(function (u) { return [esc(u.name), num(u.total), pct(u.total, total) + "%"]; })
      };
    },
    latency: function () {
      var l = data.latency;
      return {
        cols: [[bucketHeading()], ["KQL generation", 1], ["ADX execution", 1], ["LLM summary", 1]],
        rows: [["<b>Period average</b>", secs(l.avg.kql), secs(l.avg.adx), secs(l.avg.llm)]].concat(
          l.labels.map(function (lab, i) { return [lab, secs(l.kql[i]), secs(l.adx[i]), secs(l.llm[i])]; }).reverse())
      };
    },
    heatmap: function () {
      var byDay = {};
      data.heatmap.forEach(function (c) {
        var d = byDay[c[0]] || (byDay[c[0]] = { total: 0, peak: -1, peakCount: 0 });
        d.total += c[2];
        if (c[2] > d.peakCount) { d.peakCount = c[2]; d.peak = c[1]; }
      });
      return {
        cols: [["Weekday"], ["Queries", 1], ["Busiest hour", 1], ["Queries that hour", 1]],
        rows: WEEK.map(function (w) {
          var d = byDay[w[0]] || { total: 0, peak: -1, peakCount: 0 };
          return [w[1], num(d.total), d.peak >= 0 ? HOURS[d.peak] : "–", num(d.peakCount)];
        })
      };
    }
  };

  function buildTable(card) {
    var spec = TABLES[card]();
    var head = spec.cols.map(function (c) { return '<th scope="col"' + (c[1] ? ' class="num"' : "") + ">" + c[0] + "</th>"; }).join("");
    var body = spec.rows.length ? spec.rows.map(function (r) {
      return "<tr>" + r.map(function (v, i) { return "<td" + (spec.cols[i][1] ? ' class="num"' : "") + ">" + v + "</td>"; }).join("") + "</tr>";
    }).join("") : '<tr><td colspan="' + spec.cols.length + '" class="vos-empty">No data</td></tr>';
    return '<table class="vos-table"><thead><tr>' + head + "</tr></thead><tbody>" + body + "</tbody></table>";
  }

  function refreshOpenTables() {
    document.querySelectorAll("[data-card] .vos-table-wrap:not([hidden])").forEach(function (wrap) {
      wrap.innerHTML = buildTable(wrap.closest("[data-card]").dataset.card);
    });
  }

  // ── Events ─────────────────────────────────────────────────────────────────
  document.querySelectorAll("#vos-range button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      state.range = btn.dataset.range;
      if (state.range === "custom") {
        var today = dhakaToday();
        $("vos-end").value = state.end || today;
        $("vos-start").value = state.start || (data ? data.range.start : today);
        syncControls();
        return;
      }
      syncControls();
      load();
    });
  });
  $("vos-apply").addEventListener("click", function () {
    if (!$("vos-start").value || !$("vos-end").value) return;
    state.start = $("vos-start").value;
    state.end = $("vos-end").value;
    load();
  });
  $("vos-user").addEventListener("change", function () { state.user = this.value; load(); });
  $("vos-refresh").addEventListener("click", load);

  document.querySelectorAll("#vos-latest-tabs button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      latestTab = btn.dataset.latest;
      document.querySelectorAll("#vos-latest-tabs button").forEach(function (b) {
        var on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on);
      });
      if (data) renderLatest();
    });
  });

  document.querySelectorAll("[data-toggle-table]").forEach(function (btn) {
    btn.setAttribute("aria-pressed", "false");
    btn.addEventListener("click", function () {
      var card = btn.closest("[data-card]");
      var wrap = card.querySelector(".vos-table-wrap");
      var chartEl = card.querySelector(".vos-chart");
      var showTable = wrap.hidden;
      if (showTable && data) wrap.innerHTML = buildTable(card.dataset.card);
      wrap.hidden = !showTable;
      chartEl.hidden = showTable;
      btn.textContent = showTable ? "Chart" : "Table";
      btn.setAttribute("aria-pressed", String(showTable));
      if (!showTable && charts[chartEl.id]) charts[chartEl.id].resize();
    });
  });

  // Charts follow container width (window resize and Jazzmin sidebar toggle).
  var ro = new ResizeObserver(function () {
    Object.keys(charts).forEach(function (k) { charts[k].resize(); });
  });
  document.querySelectorAll(".vos-chart").forEach(function (el) { ro.observe(el); });

  // Refresh relative times / data every 5 minutes while the tab is visible.
  setInterval(function () { if (!document.hidden) load(); }, 5 * 60 * 1000);

  syncControls();
  if (state.range === "custom") { $("vos-start").value = state.start; $("vos-end").value = state.end; }
  load();
})();
