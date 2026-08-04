function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function hasNearbyAssistLead(lead, speedMps) {
  const speed = finiteNumber(speedMps, 0);
  if (speed <= 0) return false;
  const threshold = speed * 3.0;
  return Boolean(lead?.status)
    && finiteNumber(lead?.dRel, Infinity) > 0
    && finiteNumber(lead?.dRel, Infinity) < threshold;
}

const EXTERNAL_AI_MAX_AGE_MS = 750;
const EXTERNAL_AI_MAX_OBJECTS = 64;

const EXTERNAL_AI_LABELS_KO = Object.freeze({
  person: "사람",
  bicycle: "자전거",
  car: "차량",
  motorcycle: "오토바이",
  bus: "버스",
  truck: "트럭",
  "traffic light": "신호등",
  "stop sign": "정지표지판",
});

function clampUnit(value) {
  return Math.min(1, Math.max(0, finiteNumber(value, 0)));
}

function externalAIColor(className, trafficLightState) {
  if (className === "traffic light") {
    if (trafficLightState === "red") return ["rgba(255,72,72,0.12)", "rgba(255,72,72,0.96)"];
    if (trafficLightState === "yellow") return ["rgba(255,205,48,0.13)", "rgba(255,205,48,0.96)"];
    if (trafficLightState === "green") return ["rgba(44,210,105,0.12)", "rgba(44,210,105,0.96)"];
  }
  if (className === "person") return ["rgba(255,138,47,0.11)", "rgba(255,138,47,0.94)"];
  if (["car", "bus", "truck", "motorcycle", "bicycle"].includes(className)) {
    return ["rgba(54,205,255,0.10)", "rgba(54,205,255,0.94)"];
  }
  return ["rgba(255,255,255,0.08)", "rgba(255,255,255,0.90)"];
}

export function createRoadOverlayAuxRenderer(options = {}) {
  const getParams = typeof options.getParams === "function" ? options.getParams : () => ({});
  const pathZOffset = finiteNumber(options.pathZOffset, 1.22);
  const geometry = options.geometry || {};
  const ui = options.ui || {};

  if (
    typeof geometry.buildVerticalRibbon !== "function"
    || typeof geometry.drawPolygon !== "function"
    || typeof geometry.samplePathY !== "function"
    || typeof geometry.interpolate !== "function"
    || typeof geometry.projectPoint !== "function"
    || typeof geometry.drawPolyline !== "function"
    || typeof ui.getScale !== "function"
    || typeof ui.displayDistance !== "function"
    || typeof ui.clampTextAnchor !== "function"
    || typeof ui.drawText !== "function"
  ) {
    return null;
  }

  function drawBlindspotBarriers(modelPath, overlayState, hudState, calibTransform) {
    if (!modelPath || !Array.isArray(modelPath.x) || modelPath.x.length < 2) return;

    const radarState = overlayState?.radarState || {};
    const carState = hudState?.carState || {};
    const lateralPlan = overlayState?.lateralPlan || {};
    const speedMps = finiteNumber(carState?.vEgo, finiteNumber(carState?.vEgoCluster, 0));
    const laneChangeState = finiteNumber(lateralPlan?.laneChangeState, 0);
    const laneChangeDirection = finiteNumber(lateralPlan?.laneChangeDirection, 0);
    const leftBlindspot = Boolean(carState?.leftBlindspot);
    const rightBlindspot = Boolean(carState?.rightBlindspot);
    const leftAssistWarn = !leftBlindspot
      && laneChangeState === 1
      && laneChangeDirection === 1
      && hasNearbyAssistLead(radarState?.leadLeft, speedMps);
    const rightAssistWarn = !rightBlindspot
      && laneChangeState === 1
      && laneChangeDirection === 2
      && hasNearbyAssistLead(radarState?.leadRight, speedMps);
    if (!leftBlindspot && !rightBlindspot && !leftAssistWarn && !rightAssistWarn) return;

    const goldFill = "rgba(255, 215, 0, 0.48)";
    const goldStroke = "rgba(255, 215, 0, 0.84)";
    const greenFill = "rgba(0, 204, 0, 0.44)";
    const greenStroke = "rgba(0, 204, 0, 0.80)";
    const drawRibbon = (shift, fill, stroke) => {
      const ribbon = geometry.buildVerticalRibbon(calibTransform, modelPath, shift, 1.15, 0.60, 40);
      if (ribbon.length < 8) return;
      geometry.drawPolygon(ribbon, fill, stroke, 1.2);
    };

    if (leftBlindspot) drawRibbon(-1.7, goldFill, goldStroke);
    else if (leftAssistWarn) drawRibbon(-1.7, greenFill, greenStroke);

    if (rightBlindspot) drawRibbon(1.7, goldFill, goldStroke);
    else if (rightAssistWarn) drawRibbon(1.7, greenFill, greenStroke);
  }

  function drawProjectedTfMarker(modelPath, longitudinalPlan, calibTransform, videoWidth, videoHeight) {
    if (finiteNumber(getParams().ShowPathEnd, 0) <= 0) return;

    const tfDistance = finiteNumber(longitudinalPlan?.desiredDistance, 0);
    if (!Number.isFinite(tfDistance) || tfDistance <= 0) return;

    const xs = Array.isArray(modelPath?.x) ? modelPath.x : [];
    if (xs.length < 2) return;
    const lastX = finiteNumber(xs[xs.length - 1], 0);
    if (!lastX || tfDistance > lastX) return;

    const lineY = geometry.samplePathY(modelPath, tfDistance);
    const lineZ = geometry.interpolate(tfDistance, xs, Array.isArray(modelPath?.z) ? modelPath.z : []);
    if (!Number.isFinite(lineY) || !Number.isFinite(lineZ)) return;

    const left = geometry.projectPoint(calibTransform, tfDistance, lineY - 1.0, lineZ + pathZOffset);
    const right = geometry.projectPoint(calibTransform, tfDistance, lineY + 1.0, lineZ + pathZOffset);
    if (!left || !right) return;

    const uiScale = ui.getScale(videoWidth, videoHeight);
    geometry.drawPolyline([left, right], "rgba(255,255,255,0.92)", Math.max(3.0 * uiScale, 1.6));
    const labelText = `${ui.displayDistance(tfDistance).toFixed(1)}(${finiteNumber(longitudinalPlan?.tFollow, 0).toFixed(2)})`;
    const labelFontSize = Math.max(20 * uiScale, 12);
    const labelAnchor = ui.clampTextAnchor(
      { x: right.x + 10, y: right.y - 4 },
      labelText,
      labelFontSize,
      videoWidth,
      videoHeight,
    );
    ui.drawText(labelText, labelAnchor.x, labelAnchor.y, {
      fontSize: labelFontSize,
      fontWeight: 800,
      align: "left",
      strokeWidth: Math.max(3.4 * uiScale, 1.8),
    });
  }

  function drawExternalAI(phoneAIState, videoWidth, videoHeight, updatedAtMs, nowMs = Date.now()) {
    if (
      !phoneAIState?.valid
      || !phoneAIState?.connected
      || !Array.isArray(phoneAIState?.objects)
      || !Number.isFinite(Number(updatedAtMs))
      || nowMs - Number(updatedAtMs) > EXTERNAL_AI_MAX_AGE_MS
      || typeof ui.drawRoundedBox !== "function"
    ) return 0;

    const language = String(typeof ui.language === "function" ? ui.language() : "").toLowerCase();
    const isKorean = language.startsWith("ko");
    const uiScale = ui.getScale(videoWidth, videoHeight);
    const fontSize = Math.max(16 * uiScale, 11);
    const lineWidth = Math.max(2.4 * uiScale, 1.5);
    let drawn = 0;
    for (const object of phoneAIState.objects.slice(0, EXTERNAL_AI_MAX_OBJECTS)) {
      const className = String(object?.className || "").trim().toLowerCase();
      const confidence = Math.min(1, Math.max(0, finiteNumber(object?.confidence, 0)));
      const x1 = clampUnit(object?.x1) * videoWidth;
      const y1 = clampUnit(object?.y1) * videoHeight;
      const x2 = clampUnit(object?.x2) * videoWidth;
      const y2 = clampUnit(object?.y2) * videoHeight;
      if (!className || confidence <= 0 || x2 - x1 < 4 || y2 - y1 < 4) continue;

      const [fill, stroke] = externalAIColor(className, phoneAIState.trafficLightState);
      ui.drawRoundedBox(x1, y1, x2 - x1, y2 - y1, Math.max(5 * uiScale, 3), fill, stroke, lineWidth);
      const localizedName = isKorean ? (EXTERNAL_AI_LABELS_KO[className] || className) : className;
      const signal = className === "traffic light" && phoneAIState.trafficLightState && phoneAIState.trafficLightState !== "unknown"
        ? ` · ${isKorean ? ({ red: "빨강", yellow: "노랑", green: "초록" }[phoneAIState.trafficLightState] || phoneAIState.trafficLightState) : phoneAIState.trafficLightState}`
        : "";
      const trackId = Math.max(0, Math.trunc(finiteNumber(object?.trackId, 0)));
      const trackLabel = trackId > 0 ? `#${trackId} ` : "";
      const label = `${trackLabel}${localizedName}${signal} ${Math.round(confidence * 100)}%`;
      const measuredWidth = typeof ui.measureText === "function"
        ? ui.measureText(label, fontSize)
        : label.length * fontSize * 0.62;
      const labelWidth = Math.min(videoWidth - 8, Math.max(fontSize * 3.2, measuredWidth + 12 * uiScale));
      const labelHeight = Math.max(fontSize * 1.55, 20 * uiScale);
      const labelX = Math.min(Math.max(4, x1), Math.max(4, videoWidth - labelWidth - 4));
      const labelY = Math.max(4, y1 - labelHeight - 2);
      ui.drawRoundedBox(labelX, labelY, labelWidth, labelHeight, Math.max(4 * uiScale, 2), "rgba(14,18,24,0.86)", stroke, Math.max(lineWidth * 0.65, 1));
      ui.drawText(label, labelX + 6 * uiScale, labelY + labelHeight * 0.74, {
        fontSize,
        fontWeight: 800,
        align: "left",
        strokeWidth: Math.max(2.2 * uiScale, 1.2),
      });
      drawn += 1;
    }
    return drawn;
  }

  return Object.freeze({ drawBlindspotBarriers, drawProjectedTfMarker, drawExternalAI });
}

export const DriveVisionRoadOverlayAuxRenderer = Object.freeze({
  create: createRoadOverlayAuxRenderer,
});

export function installDriveVisionRoadOverlayAuxRendererFacade(target = globalThis) {
  target.DriveVisionRoadOverlayAuxRenderer = DriveVisionRoadOverlayAuxRenderer;
  return DriveVisionRoadOverlayAuxRenderer;
}
