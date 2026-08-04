import assert from "node:assert/strict";
import test from "node:test";

import { createRoadOverlayAuxRenderer } from "../src/features/drive/contents/vision/road_overlay_aux_renderer.js";

function createHarness(language = "ko-KR") {
  const boxes = [];
  const labels = [];
  const renderer = createRoadOverlayAuxRenderer({
    geometry: {
      buildVerticalRibbon: () => [],
      drawPolygon: () => {},
      samplePathY: () => 0,
      interpolate: () => 0,
      projectPoint: () => null,
      drawPolyline: () => {},
    },
    ui: {
      getScale: () => 1,
      displayDistance: (value) => value,
      clampTextAnchor: (value) => value,
      drawText: (text, x, y, options) => labels.push({ text, x, y, options }),
      drawRoundedBox: (...args) => boxes.push(args),
      measureText: (text, fontSize) => text.length * fontSize * 0.5,
      language: () => language,
    },
  });
  return { renderer, boxes, labels };
}

test("external AI draws localized object type and traffic-light color on the road video", () => {
  const { renderer, boxes, labels } = createHarness();
  const drawn = renderer.drawExternalAI({
    valid: true,
    connected: true,
    trafficLightState: "green",
    objects: [
      { className: "car", confidence: 0.88, x1: 0.1, y1: 0.2, x2: 0.4, y2: 0.7 },
      { className: "traffic light", confidence: 0.76, x1: 0.7, y1: 0.1, x2: 0.8, y2: 0.4 },
    ],
  }, 640, 360, 9_700, 10_000);

  assert.equal(drawn, 2);
  assert.equal(boxes.length, 4, "each object has a box and a label background");
  assert.equal(labels[0].text, "차량 88%");
  assert.equal(labels[1].text, "신호등 · 초록 76%");
});

test("external AI overlay rejects stale or disconnected results", () => {
  const { renderer, boxes } = createHarness("en-US");
  const state = {
    valid: true,
    connected: true,
    objects: [{ className: "person", confidence: 0.8, x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.5 }],
  };
  assert.equal(renderer.drawExternalAI(state, 640, 360, 1_000, 2_000), 0);
  assert.equal(renderer.drawExternalAI({ ...state, connected: false }, 640, 360, 1_900, 2_000), 0);
  assert.equal(boxes.length, 0);
});
