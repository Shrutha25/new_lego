// Polls /api/status (served by web_server.py, in-process with inference_loop.py)
// and drives the whole page - which of the three screens shows, the live
// camera feed, the per-step 3D model, and the step-done transition. The
// video feed itself is a separate MJPEG stream (<img src="/video_feed">)
// and needs no polling of its own.

const POLL_MS = 500;
// A step only actually changes on the page once the backend has reported
// the same next_step for this many consecutive polls - the classifier's
// per-frame confidence is noisy enough that a single stray poll can report
// a step that flips right back, and showing that instantly makes the whole
// page look like it's jumping around. Requiring 2 consecutive agreeing
// polls (~1s at POLL_MS=500) filters that out while staying fast for a
// genuine transition.
const STEP_DEBOUNCE_POLLS = 2;

let started = false; // user clicked "Start Assembly" on the home screen
let currentScreen = null;
let lastModelSrc = null;

const UNSET = Symbol("unset");
let stableNextStep = UNSET;
let pendingStepKey = null;
let pendingStepCount = 0;

let stableMismatch = false;
let pendingMismatch = null;
let pendingMismatchCount = 0;

// Frozen at the moment assembly completes - without this, the completed
// screen's "Time Taken" recomputed Date.now() - session_started_at on every
// poll and just kept counting up forever, and "Accuracy" would keep
// drifting too, since the backend keeps classifying frames (and counting
// mismatches) even after the build is done.
let completedSnapshot = null;
// Once true, completion is terminal until an explicit restart. Without
// this, a single classifier flicker on any already-confirmed tag (the same
// per-frame noise that caused the earlier wf_right/chassis flicker issues)
// makes next_step briefly leave null and come back - debounceNextStep sees
// that as a fresh transition and re-freezes completedSnapshot at a later
// timestamp, which is exactly what "Time Taken still increasing" looks
// like: not a smooth ticking clock, but jumping up each time it flickers.
let sessionCompleted = false;

function snapshotStats(data) {
  return {
    product_name: data.product_name,
    total_frames: data.total_frames,
    mismatch_frames: data.mismatch_frames,
    time_taken_s: data.session_started_at ? Date.now() / 1000 - data.session_started_at : null,
  };
}

// Same idea as debounceNextStep, for the "side incomplete" banner - it's
// driven by the same noisy per-tag confirmation, so without this it could
// flicker in and out even while the displayed step itself stays put.
function debounceMismatch(raw) {
  if (raw === stableMismatch) {
    pendingMismatch = null;
    pendingMismatchCount = 0;
    return stableMismatch;
  }
  if (raw === pendingMismatch) {
    pendingMismatchCount += 1;
  } else {
    pendingMismatch = raw;
    pendingMismatchCount = 1;
  }
  if (pendingMismatchCount >= STEP_DEBOUNCE_POLLS) {
    stableMismatch = raw;
    pendingMismatch = null;
    pendingMismatchCount = 0;
  }
  return stableMismatch;
}

function stepKeyOf(nextStep) {
  return nextStep ? String(nextStep.step) : "__complete__";
}

// Returns { value, justChanged } - value is the debounced next_step (or
// null for "assembly complete"), justChanged is true only on the poll where
// a new value was just committed (used to fire the step-done overlay once,
// not on every poll while showing the same step).
function debounceNextStep(rawNextStep) {
  const key = stepKeyOf(rawNextStep);
  const stableKey = stableNextStep === UNSET ? null : stepKeyOf(stableNextStep);

  if (key === stableKey) {
    pendingStepKey = null;
    pendingStepCount = 0;
    return { value: stableNextStep, justChanged: false };
  }

  if (key === pendingStepKey) {
    pendingStepCount += 1;
  } else {
    pendingStepKey = key;
    pendingStepCount = 1;
  }

  const isFirstEverValue = stableNextStep === UNSET;
  if (isFirstEverValue || pendingStepCount >= STEP_DEBOUNCE_POLLS) {
    stableNextStep = rawNextStep;
    pendingStepKey = null;
    pendingStepCount = 0;
    return { value: stableNextStep, justChanged: !isFirstEverValue };
  }

  // Not yet confirmed - keep showing whatever was stable before.
  return { value: stableNextStep === UNSET ? null : stableNextStep, justChanged: false };
}

function showScreen(name) {
  if (currentScreen === name) return;
  currentScreen = name;
  document.getElementById("screen-home").hidden = name !== "home";
  document.getElementById("screen-trainer").hidden = name !== "trainer";
  document.getElementById("screen-complete").hidden = name !== "complete";
}

function formatElapsed(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

async function requestReset() {
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch (err) {
    // server not reachable - nothing to do client-side, next poll will retry
  }
  // A reset should feel instant, not wait out the debounce window.
  stableNextStep = UNSET;
  pendingStepKey = null;
  pendingStepCount = 0;
  stableMismatch = false;
  pendingMismatch = null;
  pendingMismatchCount = 0;
  completedSnapshot = null;
  sessionCompleted = false;
}

document.getElementById("screen-home").addEventListener("click", () => {
  // Starting a session is exactly as much a "begin fresh" signal as the
  // restart button - without this, a tab left open on the home screen
  // (backend still running, still classifying) could accumulate stale
  // confirmations before the user ever clicks to start, and beginning
  // would show that leftover state instead of guaranteed step 1.
  started = true;
  showScreen("trainer");
  requestReset();
});
document.getElementById("restart-btn").addEventListener("click", requestReset);
document.getElementById("restart-from-complete-btn").addEventListener("click", requestReset);

document.getElementById("view-reference-btn").addEventListener("click", () => {
  document.getElementById("hint-modal").hidden = false;
});
document.getElementById("hint-modal").addEventListener("click", (e) => {
  // Close on a click anywhere except the video itself (native <video>
  // controls report the video element as the target, so they're unaffected).
  if (e.target.id !== "reference-video") {
    document.getElementById("hint-modal").hidden = true;
  }
});

// Which materials to glow at each step, so the trainee can see exactly what
// they just attached instead of having to spot it themselves on the model.
// Keyed by material *name* because model-viewer's highlight API
// (material.setEmissiveFactor) only targets whole materials, not individual
// mesh instances - found by diffing each step's materials against every
// earlier cumulative step's materials to see which ones are genuinely new.
// That diff also exposed a real limit of this model's authoring: left/right
// wheels reuse the exact same 4 materials (so this can't isolate just the
// side that's new - both sides glow together), and the roof reuses body-paint
// materials that are also used elsewhere on the chassis (so its glow spills
// onto other body panels too). Windshield is the one step with a material
// used nowhere else (Material.006, confirmed as the actual transmissive glass
// via its KHR_materials_transmission extension) - that one highlights cleanly.
const HIGHLIGHT_MATERIALS = {
  "assets/models/step_2a.gltf": ["Material.027", "Material.028", "Material.029", "Material.030"],
  "assets/models/step_2b.gltf": ["Material.027", "Material.028", "Material.029", "Material.030"],
  "assets/models/step_3.gltf": ["Material.006"],
  "assets/models/step_4.gltf": ["Material.003", "Material.004", "Material.006", "Material.007"],
};
// Warm amber glow - none of the target materials declare their own
// emissiveFactor (all default to off), and none use a baseColorTexture, so a
// flat emissive add reads clearly against their dark/flat PBR colors without
// fighting a texture.
const HIGHLIGHT_EMISSIVE = [0.9, 0.55, 0.05];

function applyHighlight() {
  const viewer = document.getElementById("viewer");
  const targets = HIGHLIGHT_MATERIALS[lastModelSrc];
  if (!targets || !viewer.model) return;
  for (const material of viewer.model.materials) {
    if (targets.includes(material.name)) {
      material.setEmissiveFactor(HIGHLIGHT_EMISSIVE);
    }
  }
}

document.getElementById("viewer").addEventListener("load", () => {
  document.getElementById("viewer-hint").hidden = true;
  // A fresh model.materials array is parsed straight from the file on every
  // src change, so there's never stale highlighting left over from a
  // previous step to clear first - only the current step's targets (if any)
  // need setting.
  applyHighlight();
});
document.getElementById("viewer").addEventListener("error", () => {
  document.getElementById("viewer-hint").hidden = false;
});

async function poll() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (res.ok) render(await res.json());
  } catch (err) {
    // server/camera not up yet - keep polling quietly rather than erroring
  }
  setTimeout(poll, POLL_MS);
}

function render(data) {
  const states = data.states || [];
  const { value: nextStep, justChanged } = debounceNextStep(data.next_step || null);

  if (!started) return; // stay on the home screen until the user clicks Start - it shows a fixed animation, not live state

  if (sessionCompleted) {
    // Terminal once reached - a stray classifier flicker on an
    // already-confirmed tag shouldn't pull the trainee back into "in
    // progress" after they've already finished. Only requestReset() clears
    // this.
    showScreen("complete");
    updateCompleteScreen();
    return;
  }

  if (!nextStep) {
    if (justChanged) {
      // Freeze the stats right now, at the actual moment of completion -
      // updateCompleteScreen() only ever reads this snapshot from here on,
      // never live data, so it stops changing once shown.
      completedSnapshot = snapshotStats(data);
      sessionCompleted = true;
      // Play the same "step is done" overlay one last time over the live
      // feed before switching to the completed screen, instead of cutting
      // straight to it.
      showStepDoneOverlay(states[states.length - 1]);
      setTimeout(() => { showScreen("complete"); updateCompleteScreen(); }, 1500);
    } else if (currentScreen !== "trainer") {
      // Not mid-transition - landed here directly (e.g. page loaded after
      // the build was already finished, so justChanged never fired). Take
      // the best approximation available now instead of leaving the stats
      // live-ticking forever.
      completedSnapshot = snapshotStats(data);
      sessionCompleted = true;
      showScreen("complete");
      updateCompleteScreen();
    }
    return;
  }

  showScreen("trainer");
  if (justChanged) {
    const idx = states.findIndex((s) => s.step === nextStep.step);
    showStepDoneOverlay(idx > 0 ? states[idx - 1] : null);
  }
  updateTrainerScreen(data, states, nextStep);
}

function showStepDoneOverlay(doneState) {
  if (!doneState) return; // nothing "completed" before the very first step
  document.getElementById("step-done-text").textContent = `${doneState.label} is done`;
  const overlay = document.getElementById("step-done-overlay");
  overlay.hidden = false;
  setTimeout(() => { overlay.hidden = true; }, 1500);
}

function updateTrainerScreen(data, states, nextStep) {
  const idx = states.findIndex((s) => s.step === nextStep.step);
  document.getElementById("step-number-tile").textContent = String(idx + 1);
  document.getElementById("step-meta").textContent = `Step ${idx + 1} of ${states.length}`;
  document.getElementById("instructions").textContent = nextStep.instructions || "";

  const mismatch = debounceMismatch(Boolean(data.mismatch));
  const alertEl = document.getElementById("mismatch-alert");
  alertEl.classList.toggle("mismatch-alert--visible", mismatch);
  if (mismatch) alertEl.textContent = data.message;

  // Uses the debounced nextStep's model, not data.model directly - data.model
  // is the raw/live backend value and updates on every poll regardless of
  // the step debounce, so using it here let the 3D model jump between steps
  // on every classifier flicker even while the step badge/instructions
  // (correctly driven by the debounced nextStep) stayed put.
  if (nextStep.model && nextStep.model !== lastModelSrc) {
    lastModelSrc = nextStep.model;
    document.getElementById("viewer").src = "/" + nextStep.model;
    document.getElementById("viewer-hint").hidden = false; // re-shown until load/error fires
  }

  const startedAt = data.session_started_at;
  const elapsed = startedAt ? Date.now() / 1000 - startedAt : 0;
  document.getElementById("time-lapsed").textContent = `Time Lapsed - ${formatElapsed(elapsed)}`;
}

function updateCompleteScreen() {
  const snap = completedSnapshot || {};
  document.getElementById("stat-product").textContent = snap.product_name || "-";

  const total = snap.total_frames || 0;
  const mismatches = snap.mismatch_frames || 0;
  const accuracy = total > 0 ? Math.round(100 * (1 - mismatches / total)) : 100;
  document.getElementById("stat-accuracy").textContent = `${accuracy}%`;

  document.getElementById("stat-time").textContent = snap.time_taken_s != null ? `${snap.time_taken_s.toFixed(1)}s` : "-";
}

// Opening the page starts a fresh session, same as pressing restart -
// without this, the backend (a single long-running inference_loop.py
// process, not restarted between browser opens) keeps whatever got
// confirmed in a previous visit, which looks identical to "jumped straight
// to a later step" even though nothing about the live evidence is wrong at
// that moment. This is a deterministic reset tied to the page loading, not
// dependent on any depth/classifier signal.
requestReset();
poll();
