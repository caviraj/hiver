/**
 * ==============================================================================
 * HIVER TELEMETRY ENGINE - HITL DYNAMICS & VALUE-PER-COST RATIO DASHBOARD
 * Milestone 7, Phase 7.1 Frontend Interactive Core
 *
 * SPECIFICATION & PRD FORMULA REFERENCE:
 *   Total Cost = (Reviewer Mins × $0.75) + Max(0, SLA Latency - 15) × $1.50
 *   Value Preserved = Severity Weight (Critical: $100, High: $50, Med: $20, Low: $5, None: $0)
 *   Value-per-Cost Ratio = Value / Max(0.01, Total Cost)
 *   Confidence Threshold: Sample Size N < 10 flagged as Low Confidence
 *
 * ICONOGRAPHY CONSTRAINT:
 *   Strictly ZERO emoji characters. All visual status iconography rendered
 *   using crisp, accessible inline SVG vectors.
 * ==============================================================================
 */

'use strict';

(function () {
  // --- CONFIGURATION CONSTANTS (Aligned with config/hitl_config.yaml) ---
  const CONFIG = {
    COST_PER_REVIEW_MINUTE: 0.75,
    SLA_PENALTY_PER_MINUTE: 1.50,
    SLA_TARGET_MINUTES: 15.0,
    MIN_CONFIDENCE_SAMPLE: 10,
    SEVERITY_WEIGHTS: {
      none: 0.0,
      low: 5.0,
      medium: 20.0,
      high: 50.0,
      critical: 100.0
    },
    TIER_NAMES: {
      tier_1: 'Frontline & General',
      tier_2: 'Technical Escalation',
      tier_3: 'Executive & Critical'
    },
    ANIMATION_DURATION: 850
  };

  // --- COMPREHENSIVE TELEMETRY DATASETS (24h, 7d, 30d) ---
  const DATASETS = {
    '24h': {
      kpis: {
        ratio: 3.18,
        total_cost: 312.75,
        labor_cost: 247.50,
        sla_cost: 65.25,
        total_value: 995.00,
        errors_caught: 15,
        sample_size: 18,
        confidence_label: 'High Confidence'
      },
      gauge: {
        surplus: 682.25,
        cost_per_error: 20.85,
        margin: '+218%',
        status: 'OPTIMAL'
      },
      series: [
        { label: '00:00', value: 70.0, cost: 21.0, ratio: 3.33 },
        { label: '04:00', value: 50.0, cost: 18.5, ratio: 2.70 },
        { label: '08:00', value: 160.0, cost: 42.0, ratio: 3.81 },
        { label: '12:00', value: 270.0, cost: 84.5, ratio: 3.19 },
        { label: '16:00', value: 295.0, cost: 91.25, ratio: 3.23 },
        { label: '20:00', value: 150.0, cost: 55.5, ratio: 2.70 }
      ],
      tiers: {
        tier_1: { events: 11, cost: 125.25, value: 430.00, ratio: 3.43, note: 'Immediate triage resolution', conf: 'N ≥ 10 Confirmed', is_low_conf: false },
        tier_2: { events: 5, cost: 112.50, value: 375.00, ratio: 3.33, note: 'Technical triage escalations', conf: 'Low Sample (N < 10)', is_low_conf: true },
        tier_3: { events: 2, cost: 75.00, value: 190.00, ratio: 2.53, note: 'Executive escalations', conf: 'Low Sample (N < 10)', is_low_conf: true }
      },
      events: [
        { id: 'EVT-9201', time: '19:42', tier: 'tier_1', review_mins: 6.0, delay_mins: 11.0, error_caught: true, severity: 'medium', cost: 4.50, value: 20.00, ratio: 4.44 },
        { id: 'EVT-9200', time: '18:15', tier: 'tier_2', review_mins: 14.0, delay_mins: 19.5, error_caught: true, severity: 'high', cost: 17.25, value: 50.00, ratio: 2.90 },
        { id: 'EVT-9199', time: '16:50', tier: 'tier_1', review_mins: 8.0, delay_mins: 13.0, error_caught: false, severity: 'none', cost: 6.00, value: 0.00, ratio: 0.00 },
        { id: 'EVT-9198', time: '15:22', tier: 'tier_3', review_mins: 22.0, delay_mins: 24.0, error_caught: true, severity: 'critical', cost: 30.00, value: 100.00, ratio: 3.33 },
        { id: 'EVT-9197', time: '13:05', tier: 'tier_1', review_mins: 7.5, delay_mins: 12.0, error_caught: true, severity: 'low', cost: 5.63, value: 5.00, ratio: 0.89 },
        { id: 'EVT-9196', time: '11:40', tier: 'tier_2', review_mins: 12.0, delay_mins: 15.0, error_caught: true, severity: 'high', cost: 9.00, value: 50.00, ratio: 5.56 }
      ]
    },
    '7d': {
      kpis: {
        ratio: 2.48,
        total_cost: 1420.50,
        labor_cost: 1025.25,
        sla_cost: 395.25,
        total_value: 3525.00,
        errors_caught: 48,
        sample_size: 64,
        confidence_label: 'High Confidence'
      },
      gauge: {
        surplus: 2104.50,
        cost_per_error: 29.59,
        margin: '+148%',
        status: 'OPTIMAL'
      },
      series: [
        { label: 'Sep 07', value: 380.0, cost: 165.0, ratio: 2.30 },
        { label: 'Sep 08', value: 440.0, cost: 182.5, ratio: 2.41 },
        { label: 'Sep 09', value: 510.0, cost: 198.0, ratio: 2.58 },
        { label: 'Sep 10', value: 580.0, cost: 215.0, ratio: 2.70 },
        { label: 'Sep 11', value: 490.0, cost: 205.0, ratio: 2.39 },
        { label: 'Sep 12', value: 575.0, cost: 228.0, ratio: 2.52 },
        { label: 'Sep 13', value: 550.0, cost: 227.0, ratio: 2.42 }
      ],
      tiers: {
        tier_1: { events: 38, cost: 570.00, value: 1680.00, ratio: 2.95, note: 'High velocity, lower error severity per ticket', conf: 'N ≥ 10 Confirmed', is_low_conf: false },
        tier_2: { events: 19, cost: 550.50, value: 1290.00, ratio: 2.34, note: 'Moderate review time with critical bug traps', conf: 'N ≥ 10 Confirmed', is_low_conf: false },
        tier_3: { events: 7, cost: 300.00, value: 555.00, ratio: 1.85, note: 'Heavy SLA penalties, highest severity weights', conf: 'Low Sample (N < 10)', is_low_conf: true }
      },
      events: [
        { id: 'EVT-8942', time: 'Sep 13 18:42', tier: 'tier_1', review_mins: 8.5, delay_mins: 12.0, error_caught: true, severity: 'medium', cost: 6.38, value: 20.00, ratio: 3.13 },
        { id: 'EVT-8941', time: 'Sep 13 17:10', tier: 'tier_2', review_mins: 16.0, delay_mins: 21.0, error_caught: true, severity: 'high', cost: 21.00, value: 50.00, ratio: 2.38 },
        { id: 'EVT-8940', time: 'Sep 13 15:35', tier: 'tier_3', review_mins: 28.0, delay_mins: 27.0, error_caught: true, severity: 'critical', cost: 39.00, value: 100.00, ratio: 2.56 },
        { id: 'EVT-8939', time: 'Sep 13 14:12', tier: 'tier_1', review_mins: 5.0, delay_mins: 10.0, error_caught: false, severity: 'none', cost: 3.75, value: 0.00, ratio: 0.00 },
        { id: 'EVT-8938', time: 'Sep 13 12:45', tier: 'tier_2', review_mins: 14.5, delay_mins: 18.0, error_caught: true, severity: 'high', cost: 15.38, value: 50.00, ratio: 3.25 },
        { id: 'EVT-8937', time: 'Sep 12 19:20', tier: 'tier_1', review_mins: 10.0, delay_mins: 16.5, error_caught: true, severity: 'low', cost: 9.75, value: 5.00, ratio: 0.51 },
        { id: 'EVT-8936', time: 'Sep 12 16:50', tier: 'tier_2', review_mins: 18.0, delay_mins: 25.0, error_caught: true, severity: 'critical', cost: 28.50, value: 100.00, ratio: 3.51 },
        { id: 'EVT-8935', time: 'Sep 12 11:30', tier: 'tier_3', review_mins: 32.0, delay_mins: 31.0, error_caught: true, severity: 'critical', cost: 48.00, value: 100.00, ratio: 2.08 }
      ]
    },
    '30d': {
      kpis: {
        ratio: 2.15,
        total_cost: 6120.75,
        labor_cost: 4410.00,
        sla_cost: 1710.75,
        total_value: 13150.00,
        errors_caught: 185,
        sample_size: 276,
        confidence_label: 'High Confidence'
      },
      gauge: {
        surplus: 7029.25,
        cost_per_error: 33.08,
        margin: '+115%',
        status: 'HEALTHY'
      },
      series: [
        { label: 'Week 1', value: 2950.0, cost: 1380.0, ratio: 2.14 },
        { label: 'Week 2', value: 3420.0, cost: 1540.0, ratio: 2.22 },
        { label: 'Week 3', value: 3180.0, cost: 1610.0, ratio: 1.98 },
        { label: 'Week 4', value: 3600.0, cost: 1590.75, ratio: 2.26 }
      ],
      tiers: {
        tier_1: { events: 168, cost: 2480.00, value: 6420.00, ratio: 2.59, note: 'Frontline volume stable across 30 days', conf: 'N ≥ 10 Confirmed', is_low_conf: false },
        tier_2: { events: 79, cost: 2310.75, value: 4890.00, ratio: 2.12, note: 'High frequency of caught API regressions', conf: 'N ≥ 10 Confirmed', is_low_conf: false },
        tier_3: { events: 29, cost: 1330.00, value: 1840.00, ratio: 1.38, note: 'Complex outages with extended SLA latencies', conf: 'N ≥ 10 Confirmed', is_low_conf: false }
      },
      events: [
        { id: 'EVT-8710', time: 'Sep 10 14:15', tier: 'tier_1', review_mins: 7.0, delay_mins: 11.0, error_caught: true, severity: 'medium', cost: 5.25, value: 20.00, ratio: 3.81 },
        { id: 'EVT-8692', time: 'Sep 08 09:30', tier: 'tier_2', review_mins: 19.0, delay_mins: 22.0, error_caught: true, severity: 'high', cost: 24.75, value: 50.00, ratio: 2.02 },
        { id: 'EVT-8645', time: 'Sep 05 16:20', tier: 'tier_3', review_mins: 34.0, delay_mins: 38.0, error_caught: true, severity: 'critical', cost: 60.00, value: 100.00, ratio: 1.67 },
        { id: 'EVT-8580', time: 'Sep 02 11:45', tier: 'tier_1', review_mins: 9.0, delay_mins: 14.0, error_caught: false, severity: 'none', cost: 6.75, value: 0.00, ratio: 0.00 },
        { id: 'EVT-8521', time: 'Aug 29 13:10', tier: 'tier_2', review_mins: 15.0, delay_mins: 17.5, error_caught: true, severity: 'medium', cost: 15.00, value: 20.00, ratio: 1.33 },
        { id: 'EVT-8490', time: 'Aug 26 10:05', tier: 'tier_3', review_mins: 40.0, delay_mins: 45.0, error_caught: true, severity: 'critical', cost: 75.00, value: 100.00, ratio: 1.33 }
      ]
    }
  };

  // --- APPLICATION STATE ---
  let currentWindow = '7d';
  let activeAnimationFrames = new Map();

  // --- UTILITY FUNCTIONS ---
  const formatCurrency = (val) => {
    return '$' + Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatNumber = (val, decimals = 2) => {
    return Number(val).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const prefersReducedMotion = () => {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  };

  // --- FLUID NUMBER COUNT-UP ENGINE WITH CUBIC EASING ---
  function animateMetric(element, targetValue, formatType = 'currency', decimals = 2) {
    if (!element) return;

    if (activeAnimationFrames.has(element)) {
      cancelAnimationFrame(activeAnimationFrames.get(element));
      activeAnimationFrames.delete(element);
    }

    if (prefersReducedMotion()) {
      if (formatType === 'currency') element.textContent = formatNumber(targetValue, decimals);
      else if (formatType === 'integer') element.textContent = Math.round(targetValue).toLocaleString('en-US');
      else element.textContent = targetValue.toFixed(decimals);
      return;
    }

    const rawCurrent = element.textContent.replace(/[^0-9.-]+/g, '');
    const startValue = parseFloat(rawCurrent) || 0;
    const startTime = performance.now();
    const duration = CONFIG.ANIMATION_DURATION;

    // Cubic bezier ease out curve
    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easeOutCubic(progress);
      const current = startValue + (targetValue - startValue) * easedProgress;

      if (formatType === 'currency') {
        element.textContent = formatNumber(current, decimals);
      } else if (formatType === 'integer') {
        element.textContent = Math.round(current).toLocaleString('en-US');
      } else {
        element.textContent = current.toFixed(decimals);
      }

      if (progress < 1) {
        const id = requestAnimationFrame(frame);
        activeAnimationFrames.set(element, id);
      } else {
        activeAnimationFrames.delete(element);
      }
    }

    const id = requestAnimationFrame(frame);
    activeAnimationFrames.set(element, id);
  }

  // --- SVG BEZIER SPLINE GENERATION ENGINE ---
  function createBezierSpline(points) {
    if (!points || points.length === 0) return '';
    if (points.length === 1) return `M ${points[0].x},${points[0].y}`;

    let path = `M ${points[0].x},${points[0].y}`;

    for (let i = 0; i < points.length - 1; i++) {
      const p0 = i > 0 ? points[i - 1] : points[i];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = i != points.length - 2 ? points[i + 2] : p2;

      const cp1x = p1.x + (p2.x - p0.x) / 6;
      const cp1y = p1.y + (p2.y - p0.y) / 6;
      const cp2x = p2.x - (p3.x - p1.x) / 6;
      const cp2y = p2.y - (p3.y - p1.y) / 6;

      path += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
    }

    return path;
  }

  // --- SVG CHART RENDERING & INTERACTIVE SCRUBBER ---
  let chartCache = {
    series: [],
    pointsVal: [],
    pointsCost: [],
    pointsRatio: []
  };

  function renderTrendChart(seriesData) {
    const valAreaPath = document.getElementById('path-value-area');
    const costAreaPath = document.getElementById('path-cost-area');
    const valLinePath = document.getElementById('path-value-line');
    const costLinePath = document.getElementById('path-cost-line');
    const ratioLinePath = document.getElementById('path-ratio-line');
    const pointsGroup = document.getElementById('chart-points');
    const xAxisContainer = document.getElementById('chart-x-labels');

    if (!valAreaPath || !seriesData || seriesData.length === 0) return;

    // Viewport dimensions
    const width = 700;
    const height = 260;
    const paddingLeft = 55;
    const paddingRight = 35;
    const paddingTop = 35;
    const paddingBottom = 30;

    const usableWidth = width - paddingLeft - paddingRight;
    const usableHeight = height - paddingTop - paddingBottom;
    const baselineY = height - paddingBottom;

    // Determine scale maxima
    let maxAmount = 100;
    let maxRatio = 4.0;

    seriesData.forEach(d => {
      if (d.value > maxAmount) maxAmount = d.value;
      if (d.cost > maxAmount) maxAmount = d.cost;
      if (d.ratio > maxRatio) maxRatio = d.ratio;
    });

    maxAmount *= 1.15; // 15% headroom
    maxRatio *= 1.10;

    // Calculate normalized point coordinates
    const numPoints = seriesData.length;
    const stepX = usableWidth / (numPoints - 1);

    const pointsVal = [];
    const pointsCost = [];
    const pointsRatio = [];

    seriesData.forEach((d, i) => {
      const x = paddingLeft + i * stepX;
      const yVal = baselineY - (d.value / maxAmount) * usableHeight;
      const yCost = baselineY - (d.cost / maxAmount) * usableHeight;
      const yRatio = baselineY - (d.ratio / maxRatio) * usableHeight;

      pointsVal.push({ x, y: yVal, data: d });
      pointsCost.push({ x, y: yCost, data: d });
      pointsRatio.push({ x, y: yRatio, data: d });
    });

    chartCache = {
      series: seriesData,
      pointsVal,
      pointsCost,
      pointsRatio,
      baselineY
    };

    // Build cubic bezier lines
    const lineValD = createBezierSpline(pointsVal);
    const lineCostD = createBezierSpline(pointsCost);
    const lineRatioD = createBezierSpline(pointsRatio);

    // Build enclosed area paths
    const lastVal = pointsVal[pointsVal.length - 1];
    const firstVal = pointsVal[0];
    const areaValD = `${lineValD} L ${lastVal.x.toFixed(1)},${baselineY} L ${firstVal.x.toFixed(1)},${baselineY} Z`;

    const lastCost = pointsCost[pointsCost.length - 1];
    const firstCost = pointsCost[0];
    const areaCostD = `${lineCostD} L ${lastCost.x.toFixed(1)},${baselineY} L ${firstCost.x.toFixed(1)},${baselineY} Z`;

    // Apply path descriptions
    valLinePath.setAttribute('d', lineValD);
    costLinePath.setAttribute('d', lineCostD);
    ratioLinePath.setAttribute('d', lineRatioD);
    valAreaPath.setAttribute('d', areaValD);
    costAreaPath.setAttribute('d', areaCostD);

    // Render interactive data point dots
    pointsGroup.innerHTML = '';
    pointsVal.forEach((pt, i) => {
      const costPt = pointsCost[i];

      // Value dot
      const circleVal = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circleVal.setAttribute('cx', pt.x.toFixed(1));
      circleVal.setAttribute('cy', pt.y.toFixed(1));
      circleVal.setAttribute('r', '3.5');
      circleVal.setAttribute('class', 'chart-dot dot-emerald');
      pointsGroup.appendChild(circleVal);

      // Cost dot
      const circleCost = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circleCost.setAttribute('cx', costPt.x.toFixed(1));
      circleCost.setAttribute('cy', costPt.y.toFixed(1));
      circleCost.setAttribute('r', '3.5');
      circleCost.setAttribute('class', 'chart-dot dot-rose');
      pointsGroup.appendChild(circleCost);
    });

    // Populate X Axis Labels
    if (xAxisContainer) {
      xAxisContainer.innerHTML = '';
      seriesData.forEach(d => {
        const span = document.createElement('span');
        span.className = 'x-axis-label';
        span.textContent = d.label;
        xAxisContainer.appendChild(span);
      });
    }
  }

  // --- SETUP CHART TOOLTIP & SCRUBBER LISTENER ---
  function setupChartScrubber() {
    const chartWrapper = document.getElementById('chart-wrapper');
    const svg = document.getElementById('trend-chart');
    const tooltip = document.getElementById('chart-tooltip');
    const scrubberLine = document.getElementById('scrubber-line');
    const scrubberDotVal = document.getElementById('scrubber-dot-val');
    const scrubberDotCost = document.getElementById('scrubber-dot-cost');

    const tooltipDate = document.getElementById('tooltip-date');
    const tooltipVal = document.getElementById('tooltip-val');
    const tooltipCost = document.getElementById('tooltip-cost');
    const tooltipRatio = document.getElementById('tooltip-ratio');

    if (!chartWrapper || !svg) return;

    function handleScrubber(clientX) {
      if (!chartCache.pointsVal || chartCache.pointsVal.length === 0) return;

      const rect = svg.getBoundingClientRect();
      const relativeX = clientX - rect.left;
      const svgScaleX = 700 / rect.width;
      const targetSvgX = relativeX * svgScaleX;

      // Find nearest point
      let nearestIdx = 0;
      let minDiff = Infinity;

      chartCache.pointsVal.forEach((pt, idx) => {
        const diff = Math.abs(pt.x - targetSvgX);
        if (diff < minDiff) {
          minDiff = diff;
          nearestIdx = idx;
        }
      });

      const ptVal = chartCache.pointsVal[nearestIdx];
      const ptCost = chartCache.pointsCost[nearestIdx];
      const data = chartCache.series[nearestIdx];

      if (!ptVal || !data) return;

      // Update scrubber elements
      scrubberLine.setAttribute('x1', ptVal.x.toFixed(1));
      scrubberLine.setAttribute('x2', ptVal.x.toFixed(1));
      scrubberLine.setAttribute('opacity', '1');

      scrubberDotVal.setAttribute('cx', ptVal.x.toFixed(1));
      scrubberDotVal.setAttribute('cy', ptVal.y.toFixed(1));
      scrubberDotVal.setAttribute('opacity', '1');

      scrubberDotCost.setAttribute('cx', ptCost.x.toFixed(1));
      scrubberDotCost.setAttribute('cy', ptCost.y.toFixed(1));
      scrubberDotCost.setAttribute('opacity', '1');

      // Update tooltip content
      tooltipDate.textContent = data.label;
      tooltipVal.textContent = formatCurrency(data.value);
      tooltipCost.textContent = formatCurrency(data.cost);
      tooltipRatio.textContent = data.ratio.toFixed(2) + 'x';

      // Position tooltip cleanly
      const tooltipSvgX = (ptVal.x / 700) * rect.width;
      const tooltipLeft = Math.max(10, Math.min(rect.width - 150, tooltipSvgX - 70));
      tooltip.style.left = `${tooltipLeft}px`;
      tooltip.style.top = '12px';
      tooltip.classList.add('visible');
      tooltip.setAttribute('aria-hidden', 'false');
    }

    function hideScrubber() {
      if (scrubberLine) scrubberLine.setAttribute('opacity', '0');
      if (scrubberDotVal) scrubberDotVal.setAttribute('opacity', '0');
      if (scrubberDotCost) scrubberDotCost.setAttribute('opacity', '0');
      if (tooltip) {
        tooltip.classList.remove('visible');
        tooltip.setAttribute('aria-hidden', 'true');
      }
    }

    svg.addEventListener('mousemove', (e) => handleScrubber(e.clientX));
    svg.addEventListener('mouseleave', hideScrubber);

    svg.addEventListener('touchmove', (e) => {
      if (e.touches.length > 0) handleScrubber(e.touches[0].clientX);
    }, { passive: true });
    svg.addEventListener('touchend', hideScrubber);
  }

  // --- RADIAL EFFICIENCY GAUGE ENGINE ---
  function updateRadialGauge(ratio, gaugeMeta) {
    const gaugeFill = document.getElementById('radial-gauge-fill');
    const gaugeDisplayNum = document.getElementById('gauge-display-num');
    const gaugeStatusBadge = document.getElementById('gauge-status-badge');
    const gaugeSurplus = document.getElementById('gauge-surplus');
    const gaugeCostPerErr = document.getElementById('gauge-cost-per-err');
    const gaugeMargin = document.getElementById('gauge-margin');

    if (!gaugeFill) return;

    // Circumference for r=78 is ~490.088
    const totalCircumference = 490.088;
    const maxRatio = 4.0;
    const clampedRatio = Math.max(0, Math.min(ratio, maxRatio));
    const offset = totalCircumference - (clampedRatio / maxRatio) * totalCircumference;

    gaugeFill.style.strokeDashoffset = offset.toFixed(1);

    // Color stroke according to health threshold
    if (ratio >= 2.0) {
      gaugeFill.style.stroke = 'var(--emerald-500)';
    } else if (ratio >= 1.2) {
      gaugeFill.style.stroke = 'var(--cyan-400)';
    } else if (ratio >= 1.0) {
      gaugeFill.style.stroke = 'var(--amber-400)';
    } else {
      gaugeFill.style.stroke = 'var(--rose-500)';
    }

    animateMetric(gaugeDisplayNum, ratio, 'decimal', 2);

    if (gaugeStatusBadge) {
      gaugeStatusBadge.textContent = gaugeMeta.status;
      gaugeStatusBadge.className = 'gauge-status';
      if (gaugeMeta.status === 'OPTIMAL') gaugeStatusBadge.classList.add('status-emerald');
      else if (gaugeMeta.status === 'HEALTHY') gaugeStatusBadge.classList.add('status-cyan');
      else if (gaugeMeta.status === 'MARGINAL') gaugeStatusBadge.classList.add('status-amber');
      else gaugeStatusBadge.classList.add('status-rose');
    }

    if (gaugeSurplus) {
      const prefix = gaugeMeta.surplus >= 0 ? '+$' : '-$';
      gaugeSurplus.textContent = prefix + Math.abs(gaugeMeta.surplus).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    if (gaugeCostPerErr) {
      gaugeCostPerErr.textContent = formatCurrency(gaugeMeta.cost_per_error);
    }
    if (gaugeMargin) {
      gaugeMargin.textContent = gaugeMeta.margin;
    }
  }

  // --- TIER DISSECTION CARDS UPDATE ---
  function updateTierCards(tiers) {
    ['tier_1', 'tier_2', 'tier_3'].forEach((tierKey, index) => {
      const tierNum = index + 1;
      const data = tiers[tierKey];
      if (!data) return;

      const eventsEl = document.getElementById(`t${tierNum}-events`);
      const costEl = document.getElementById(`t${tierNum}-cost`);
      const valEl = document.getElementById(`t${tierNum}-val`);
      const ratioBadge = document.getElementById(`t${tierNum}-ratio-badge`);
      const confBadge = document.getElementById(`t${tierNum}-conf`);
      const noteEl = document.getElementById(`t${tierNum}-note`);

      if (eventsEl) animateMetric(eventsEl, data.events, 'integer');
      if (costEl) animateMetric(costEl, data.cost, 'currency');
      if (valEl) animateMetric(valEl, data.value, 'currency');

      if (ratioBadge) {
        ratioBadge.textContent = `${data.ratio.toFixed(2)}x Ratio`;
        ratioBadge.className = 'tier-ratio-badge ' + (data.ratio >= 2.0 ? 'badge-emerald' : data.ratio >= 1.0 ? 'badge-amber' : 'badge-rose');
      }

      if (confBadge) {
        confBadge.textContent = data.conf;
        confBadge.className = 'confidence-pill ' + (data.is_low_conf ? 'pill-amber' : 'pill-green');
      }

      if (noteEl && data.note) {
        noteEl.textContent = data.note;
      }
    });
  }

  // --- AUDIT LOG EVENT STREAM RENDERING ---
  function renderAuditTable(events) {
    const tbody = document.getElementById('events-tbody');
    const filterSelect = document.getElementById('filter-tier');
    if (!tbody || !events) return;

    const activeFilter = filterSelect ? filterSelect.value : 'all';

    tbody.innerHTML = '';

    const filtered = events.filter(evt => {
      if (activeFilter === 'all') return true;
      return evt.tier === activeFilter;
    });

    if (filtered.length === 0) {
      const emptyRow = document.createElement('tr');
      emptyRow.innerHTML = `
        <td colspan="10" style="text-align:center; padding: 2rem; color: var(--text-muted);">
          No audit events found for the selected tier filter.
        </td>
      `;
      tbody.appendChild(emptyRow);
      return;
    }

    filtered.forEach(evt => {
      const tr = createEventRow(evt);
      tbody.appendChild(tr);
    });
  }

  function createEventRow(evt) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-tier', evt.tier);

    // Tier badge mapping
    const tierPillClass = evt.tier === 'tier_1' ? 'tier-pill-1' : evt.tier === 'tier_2' ? 'tier-pill-2' : 'tier-pill-3';
    const tierLabel = evt.tier === 'tier_1' ? 'TIER 1' : evt.tier === 'tier_2' ? 'TIER 2' : 'TIER 3';

    // Interception SVG icon (No emojis!)
    const errorCaughtHtml = evt.error_caught
      ? `<span class="icon-text text-emerald" title="Error Successfully Caught">
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>
           <span>Caught</span>
         </span>`
      : `<span class="icon-text text-muted" title="Clean ticket, no error">
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
           <span>Clean</span>
         </span>`;

    // Severity badge mapping
    let severityBadgeClass = 'badge-muted';
    let severityName = 'None';
    if (evt.severity === 'critical') { severityBadgeClass = 'badge-rose'; severityName = 'Critical'; }
    else if (evt.severity === 'high') { severityBadgeClass = 'badge-amber'; severityName = 'High'; }
    else if (evt.severity === 'medium') { severityBadgeClass = 'badge-indigo'; severityName = 'Medium'; }
    else if (evt.severity === 'low') { severityBadgeClass = 'badge-cyan'; severityName = 'Low'; }

    // Ratio badge color
    const ratioBadgeClass = evt.ratio >= 2.0 ? 'badge-emerald' : evt.ratio >= 1.0 ? 'badge-cyan' : evt.ratio > 0 ? 'badge-amber' : 'badge-muted';

    tr.innerHTML = `
      <td class="fira-num text-cyan">${evt.id}</td>
      <td class="text-muted">${evt.time}</td>
      <td><span class="tier-pill ${tierPillClass}">${tierLabel}</span></td>
      <td class="fira-num">${evt.review_mins.toFixed(1)}m</td>
      <td class="fira-num">${evt.delay_mins.toFixed(1)}m</td>
      <td>${errorCaughtHtml}</td>
      <td><span class="badge ${severityBadgeClass}">${severityName}</span></td>
      <td class="fira-num">${formatCurrency(evt.cost)}</td>
      <td class="fira-num text-emerald">${formatCurrency(evt.value)}</td>
      <td><span class="badge ${ratioBadgeClass} fira-num">${evt.ratio.toFixed(2)}x</span></td>
    `;

    return tr;
  }

  // --- APPLY WINDOW REFRESH (KPIs, Chart, Gauge, Tiers, Table) ---
  function applyWindowData(windowKey) {
    currentWindow = windowKey;
    const dataset = DATASETS[windowKey];
    if (!dataset) return;

    // 1. KPI Cards Count-up
    const kpiRatio = document.getElementById('kpi-ratio-value');
    const kpiCost = document.getElementById('kpi-cost-value');
    const kpiValue = document.getElementById('kpi-value-value');
    const kpiReviews = document.getElementById('kpi-reviews-value');

    const kpiRatioCents = document.getElementById('kpi-ratio-cents');
    const ratioFillBar = document.getElementById('ratio-fill-bar');
    const ratioVerdictBadge = document.getElementById('ratio-verdict-badge');

    const kpiCostLabor = document.getElementById('kpi-cost-labor');
    const kpiCostSla = document.getElementById('kpi-cost-sla');
    const kpiErrorsCaught = document.getElementById('kpi-errors-caught');

    const kpiSampleSize = document.getElementById('kpi-sample-size');
    const kpiSampleBar = document.getElementById('kpi-sample-bar');
    const kpiConfidenceBadge = document.getElementById('kpi-confidence-badge');

    animateMetric(kpiRatio, dataset.kpis.ratio, 'decimal', 2);
    animateMetric(kpiCost, dataset.kpis.total_cost, 'currency', 2);
    animateMetric(kpiValue, dataset.kpis.total_value, 'currency', 2);
    animateMetric(kpiReviews, dataset.kpis.sample_size, 'integer');

    if (kpiRatioCents) kpiRatioCents.textContent = `$${dataset.kpis.ratio.toFixed(2)}`;
    if (ratioFillBar) {
      const fillPct = Math.min(100, Math.max(10, (dataset.kpis.ratio / 3.2) * 100));
      ratioFillBar.style.width = `${fillPct}%`;
    }
    if (ratioVerdictBadge) {
      if (dataset.kpis.ratio >= 2.0) {
        ratioVerdictBadge.textContent = 'High ROI';
        ratioVerdictBadge.className = 'badge badge-emerald';
      } else if (dataset.kpis.ratio >= 1.0) {
        ratioVerdictBadge.textContent = 'Profitable';
        ratioVerdictBadge.className = 'badge badge-cyan';
      } else {
        ratioVerdictBadge.textContent = 'Deficit';
        ratioVerdictBadge.className = 'badge badge-rose';
      }
    }

    if (kpiCostLabor) kpiCostLabor.textContent = formatCurrency(dataset.kpis.labor_cost);
    if (kpiCostSla) kpiCostSla.textContent = formatCurrency(dataset.kpis.sla_cost);
    if (kpiErrorsCaught) animateMetric(kpiErrorsCaught, dataset.kpis.errors_caught, 'integer');

    if (kpiSampleSize) kpiSampleSize.textContent = dataset.kpis.sample_size;
    if (kpiSampleBar) {
      const samplePct = Math.min(100, (dataset.kpis.sample_size / CONFIG.MIN_CONFIDENCE_SAMPLE) * 100);
      kpiSampleBar.style.width = `${samplePct}%`;
    }
    if (kpiConfidenceBadge) {
      if (dataset.kpis.sample_size >= CONFIG.MIN_CONFIDENCE_SAMPLE) {
        kpiConfidenceBadge.textContent = 'High Confidence';
        kpiConfidenceBadge.className = 'badge badge-indigo';
      } else {
        kpiConfidenceBadge.textContent = 'Low Sample (N < 10)';
        kpiConfidenceBadge.className = 'badge badge-amber';
      }
    }

    // 2. Render Time Series Bezier Chart
    renderTrendChart(dataset.series);

    // 3. Update Radial Gauge
    updateRadialGauge(dataset.kpis.ratio, dataset.gauge);

    // 4. Update Tier Breakdown Cards
    updateTierCards(dataset.tiers);

    // 5. Update Audit Table
    renderAuditTable(dataset.events);
  }

  // --- INTERACTIVE HITL SCENARIO CALCULATOR ENGINE ---
  function setupSimulator() {
    const minutesSlider = document.getElementById('sim-minutes-slider');
    const delaySlider = document.getElementById('sim-delay-slider');
    const severitySelect = document.getElementById('sim-severity-select');
    const resetBtn = document.getElementById('sim-reset-btn');

    const minutesDisplay = document.getElementById('sim-minutes-display');
    const delayDisplay = document.getElementById('sim-delay-display');
    const severityDisplay = document.getElementById('sim-value-weight-display');

    const resultRatio = document.getElementById('sim-result-ratio');
    const verdictTag = document.getElementById('sim-verdict-tag');
    const laborCostEl = document.getElementById('sim-labor-cost');
    const slaPenaltyEl = document.getElementById('sim-sla-penalty');
    const totalCostEl = document.getElementById('sim-total-cost');
    const totalValEl = document.getElementById('sim-total-val');
    const roiPercentEl = document.getElementById('sim-roi-percent');
    const barFillEl = document.getElementById('sim-bar-fill');

    if (!minutesSlider || !delaySlider || !severitySelect) return;

    function recalculateSimulator() {
      const minutes = parseFloat(minutesSlider.value);
      const delay = parseFloat(delaySlider.value);
      const severity = severitySelect.value;
      const severityWeight = CONFIG.SEVERITY_WEIGHTS[severity] || 0.0;

      // Update Slider Labels
      minutesDisplay.textContent = `${minutes} mins`;
      delayDisplay.textContent = `${delay} mins (Target 15)`;
      severityDisplay.textContent = `$${severityWeight.toFixed(2)} Weight`;

      // PRD Formula Computations
      const laborCost = minutes * CONFIG.COST_PER_REVIEW_MINUTE;
      const penaltyMinutes = Math.max(0, delay - CONFIG.SLA_TARGET_MINUTES);
      const slaPenalty = penaltyMinutes * CONFIG.SLA_PENALTY_PER_MINUTE;
      const totalCost = laborCost + slaPenalty;

      const effectiveCost = totalCost > 0 ? totalCost : 0.01;
      const ratio = severityWeight / effectiveCost;
      const surplus = severityWeight - totalCost;
      const surplusPercent = totalCost > 0 ? ((surplus / totalCost) * 100) : 0;

      // Output values
      resultRatio.textContent = `${ratio.toFixed(2)}x`;
      laborCostEl.textContent = formatCurrency(laborCost);
      slaPenaltyEl.textContent = `+${formatCurrency(slaPenalty)}`;
      totalCostEl.textContent = formatCurrency(totalCost);
      totalValEl.textContent = formatCurrency(severityWeight);

      // ROI badge verdict & color styling
      if (ratio >= 2.0) {
        verdictTag.textContent = 'High Value Justified';
        verdictTag.className = 'badge badge-emerald';
        resultRatio.style.color = 'var(--emerald-400)';
      } else if (ratio >= 1.0) {
        verdictTag.textContent = 'Value Positive';
        verdictTag.className = 'badge badge-cyan';
        resultRatio.style.color = 'var(--cyan-400)';
      } else if (ratio >= 0.5) {
        verdictTag.textContent = 'Marginal Justification';
        verdictTag.className = 'badge badge-amber';
        resultRatio.style.color = 'var(--amber-400)';
      } else {
        verdictTag.textContent = 'Cost Deficit';
        verdictTag.className = 'badge badge-rose';
        resultRatio.style.color = 'var(--rose-400)';
      }

      // Surplus badge
      const prefix = surplus >= 0 ? '+' : '';
      roiPercentEl.textContent = `${prefix}${surplusPercent.toFixed(0)}% Surplus`;
      if (surplus >= 0) roiPercentEl.className = 'fira-num text-emerald';
      else roiPercentEl.className = 'fira-num text-rose';

      // Yield Progress Fill
      const clampedFill = Math.min(100, Math.max(6, (ratio / 4.0) * 100));
      barFillEl.style.width = `${clampedFill}%`;
      if (ratio >= 1.5) barFillEl.style.background = 'var(--emerald-500)';
      else if (ratio >= 1.0) barFillEl.style.background = 'var(--cyan-400)';
      else barFillEl.style.background = 'var(--rose-500)';
    }

    minutesSlider.addEventListener('input', recalculateSimulator);
    delaySlider.addEventListener('input', recalculateSimulator);
    severitySelect.addEventListener('change', recalculateSimulator);

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        minutesSlider.value = '12';
        delaySlider.value = '18';
        severitySelect.value = 'high';
        recalculateSimulator();
        showToast('Simulator parameters reset to baseline defaults.', 'info');
      });
    }

    recalculateSimulator();
  }

  // --- TOAST NOTIFICATION SYSTEM (NO EMOJIS, PURE SVG) ---
  function showToast(message, type = 'success', subtitle = '') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `glass-toast toast-${type}`;

    let iconSvg = '';
    if (type === 'success') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else if (type === 'warning') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    } else {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
    }

    toast.innerHTML = `
      <div class="toast-icon">${iconSvg}</div>
      <div class="toast-content">
        <div class="toast-title">${message}</div>
        ${subtitle ? `<div class="toast-sub">${subtitle}</div>` : ''}
      </div>
      <button type="button" class="toast-close" aria-label="Dismiss Notification">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
    `;

    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => dismissToast(toast));

    container.appendChild(toast);

    // Auto dismiss after 4.5 seconds
    setTimeout(() => {
      dismissToast(toast);
    }, 4500);
  }

  function dismissToast(toast) {
    if (!toast || toast.classList.contains('dismissing')) return;
    toast.classList.add('dismissing');
    toast.addEventListener('transitionend', () => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    });
  }

  // --- LIVE EVENT SIMULATION INJECTION ---
  function setupEventSimulation() {
    const simBtn = document.getElementById('simulate-event-btn');
    if (!simBtn) return;

    simBtn.addEventListener('click', () => {
      // Pick random tier (weighted)
      const rand = Math.random();
      const tier = rand < 0.55 ? 'tier_1' : rand < 0.85 ? 'tier_2' : 'tier_3';

      // Duration & Delay bounds per tier
      let reviewMins = 0;
      let delayMins = 0;
      if (tier === 'tier_1') {
        reviewMins = +(4.0 + Math.random() * 8.0).toFixed(1);
        delayMins = +(8.0 + Math.random() * 8.0).toFixed(1);
      } else if (tier === 'tier_2') {
        reviewMins = +(10.0 + Math.random() * 12.0).toFixed(1);
        delayMins = +(14.0 + Math.random() * 10.0).toFixed(1);
      } else {
        reviewMins = +(20.0 + Math.random() * 18.0).toFixed(1);
        delayMins = +(22.0 + Math.random() * 16.0).toFixed(1);
      }

      // Caught Error Severity
      const catchesError = Math.random() > 0.25; // 75% error catch probability
      let severity = 'none';
      if (catchesError) {
        if (tier === 'tier_3') severity = Math.random() > 0.3 ? 'critical' : 'high';
        else if (tier === 'tier_2') severity = Math.random() > 0.4 ? 'high' : 'medium';
        else severity = Math.random() > 0.5 ? 'medium' : 'low';
      }

      const severityVal = CONFIG.SEVERITY_WEIGHTS[severity];
      const laborCost = reviewMins * CONFIG.COST_PER_REVIEW_MINUTE;
      const penaltyMins = Math.max(0, delayMins - CONFIG.SLA_TARGET_MINUTES);
      const slaCost = penaltyMins * CONFIG.SLA_PENALTY_PER_MINUTE;
      const totalCost = laborCost + slaCost;
      const ratio = severityVal / (totalCost > 0 ? totalCost : 0.01);

      const eventId = 'EVT-' + Math.floor(1000 + Math.random() * 9000);
      const now = new Date();
      const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

      const newEvent = {
        id: eventId,
        time: timeStr,
        tier: tier,
        review_mins: reviewMins,
        delay_mins: delayMins,
        error_caught: catchesError,
        severity: severity,
        cost: totalCost,
        value: severityVal,
        ratio: ratio
      };

      // Add to current window dataset
      const activeData = DATASETS[currentWindow];
      if (activeData) {
        activeData.events.unshift(newEvent);
        activeData.kpis.total_cost += totalCost;
        activeData.kpis.labor_cost += laborCost;
        activeData.kpis.sla_cost += slaCost;
        activeData.kpis.total_value += severityVal;
        activeData.kpis.sample_size += 1;
        if (catchesError && (severity === 'critical' || severity === 'high')) {
          activeData.kpis.errors_caught += 1;
        }
        activeData.kpis.ratio = activeData.kpis.total_value / activeData.kpis.total_cost;

        // Update Tier Data
        const tierStat = activeData.tiers[tier];
        if (tierStat) {
          tierStat.events += 1;
          tierStat.cost += totalCost;
          tierStat.value += severityVal;
          tierStat.ratio = tierStat.value / (tierStat.cost || 0.01);
          if (tierStat.events >= 10) {
            tierStat.conf = 'N ≥ 10 Confirmed';
            tierStat.is_low_conf = false;
          }
        }

        // Update Gauge
        activeData.gauge.surplus = activeData.kpis.total_value - activeData.kpis.total_cost;
        activeData.gauge.cost_per_error = activeData.kpis.total_cost / (activeData.kpis.errors_caught || 1);
        activeData.gauge.margin = `${((activeData.kpis.ratio - 1) * 100).toFixed(0)}%`;
        activeData.gauge.status = activeData.kpis.ratio >= 2.0 ? 'OPTIMAL' : activeData.kpis.ratio >= 1.2 ? 'HEALTHY' : activeData.kpis.ratio >= 1.0 ? 'MARGINAL' : 'DEFICIT';

        // Animate updated KPIs
        applyWindowData(currentWindow);

        // Prepend and animate table row
        const tbody = document.getElementById('events-tbody');
        if (tbody) {
          const row = createEventRow(newEvent);
          row.classList.add('new-event-row');
          tbody.insertBefore(row, tbody.firstChild);
        }

        // Trigger toast
        const tierTitle = CONFIG.TIER_NAMES[tier];
        const toastType = ratio >= 1.0 ? 'success' : 'warning';
        showToast(
          `New Review Ingested: ${eventId}`,
          toastType,
          `${tierTitle} | Value: ${formatCurrency(severityVal)} | Cost: ${formatCurrency(totalCost)} | Ratio: ${ratio.toFixed(2)}x`
        );
      }
    });
  }

  // --- WINDOW SWITCHER CONTROLLER ---
  function setupWindowSwitcher() {
    const buttons = document.querySelectorAll('.window-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetWindow = btn.getAttribute('data-window');
        if (targetWindow === currentWindow) return;

        buttons.forEach(b => {
          b.classList.remove('active');
          b.setAttribute('aria-pressed', 'false');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');

        applyWindowData(targetWindow);
        showToast(`Switched telemetry window to ${targetWindow.toUpperCase()}`, 'info');
      });
    });
  }

  // --- TABLE TIER FILTER CONTROLLER ---
  function setupTableFilter() {
    const filterSelect = document.getElementById('filter-tier');
    if (!filterSelect) return;

    filterSelect.addEventListener('change', () => {
      const activeData = DATASETS[currentWindow];
      if (activeData) {
        renderAuditTable(activeData.events);
      }
    });
  }

  // --- INITIALIZATION ON DOM READY ---
  function init() {
    setupWindowSwitcher();
    setupChartScrubber();
    setupSimulator();
    setupEventSimulation();
    setupTableFilter();

    // Initial render with default 7d window
    applyWindowData('7d');

    // Welcome notice
    setTimeout(() => {
      showToast('Telemetry Stream Active', 'info', 'Connected to Hiver M7.P7.1 metrics pipeline.');
    }, 600);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
