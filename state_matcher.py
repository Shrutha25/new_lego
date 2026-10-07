"""Rule-based assembly state matcher (Section 4) with temporal smoothing (Section 8).

Tracks confirmed tag state across a whole session, not per-frame. A tag can
un-confirm, but only when there's real evidence the part was removed rather
than just rotated out of view - see _maybe_unconfirm() below for how that
distinction is made. This was removed for a while (Issue #8, 2026-09-11)
because the classifier was unreliable enough that un-confirming caused more
harm than it prevented - genuinely-attached wheels reading absent for long
enough to un-confirm, sending the trainee back to redo a finished step.
Reinstated (Issue #15, 2026-09-11) to test whether the classifier is
reliable enough for this now that the worst confirmed shortcuts (chassis
implies wheels, right implies left) have been retrained out. If wheels
start falsely un-confirming again, that's the signal this needs to go back
to one-way rather than being tuned further - see UNCONFIRM_WINDOW_FRAMES /
UNCONFIRM_MAX_FRACTION in config.py.

Guidance is derived from the furthest step whose requirements are all
confirmed.

Confirmation uses a sliding-window majority vote rather than a strict run of
consecutive above-threshold frames: real camera/model output is noisy, so a
genuinely-attached part can dip below threshold for a stray frame here and
there, and a strict consecutive-run requirement wipes out all progress on
any single dip - in practice this can mean nothing ever stabilizes. A
majority vote tolerates that noise. It also closes a second problem with a
short consecutive-run requirement: a brief false-positive streak (a
lighting flicker, a shadow) is just as easy to satisfy as genuine
attachment - requiring sustained evidence across a longer window makes a
short spurious streak insufficient to lock in a wrong confirmation.
"""
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import config


@dataclass
class Guidance:
    session_state: Dict[str, bool]
    completed_step: object
    next_step: Optional[dict]
    message: str
    instructions: Optional[str] = None
    mismatch: bool = False


class StateMatcher:
    def __init__(self, states_path=config.STATES_PATH, tags=None,
                 threshold=config.CONFIDENCE_THRESHOLD,
                 window_frames=config.CONFIRMATION_WINDOW_FRAMES,
                 min_true_fraction=config.CONFIRMATION_MIN_FRACTION,
                 unconfirm_window_frames=config.UNCONFIRM_WINDOW_FRAMES,
                 unconfirm_max_fraction=config.UNCONFIRM_MAX_FRACTION,
                 empty_reset_frames=config.EMPTY_RESET_FRAMES,
                 side_groups=None):
        with open(states_path) as f:
            self.states: List[dict] = json.load(f)
        self.tags = list(tags) if tags else list(config.TAGS)
        self.threshold = threshold
        self.window_frames = window_frames
        self.min_true_fraction = min_true_fraction
        self.unconfirm_window_frames = unconfirm_window_frames
        self.unconfirm_max_fraction = unconfirm_max_fraction
        self.empty_reset_frames = empty_reset_frames
        self.side_groups = side_groups or config.SIDE_GROUPS
        self._tag_side = {t: side for side, side_tags in self.side_groups.items() for t in side_tags}
        self._paired_tags = set(self._tag_side)  # tags that belong to a side_groups pair
        self.reset()

    def reset(self):
        """Start a fresh session: all tags unconfirmed."""
        self.session_state = {t: False for t in self.tags}
        self._recent = {t: deque(maxlen=self.window_frames) for t in self.tags}
        self._recent_long = {t: deque(maxlen=self.unconfirm_window_frames) for t in self.tags}
        self._recent_presence = deque(maxlen=self.unconfirm_window_frames)
        self._empty_streak = 0

    def _maybe_unconfirm(self, tag, sides_in_view):
        """Un-confirm `tag` if there's sustained evidence it was removed, not just rotated
        out of view (for a wheel) or just poorly seen while the trainee's hands are actively
        working on the car (for anything else).

        A removed wheel and a rotated-away wheel read identically (False) on their own, so a
        wheel only un-confirms while its *side* is currently in view per this frame's live
        readings (sides_in_view) - proof we're actually looking at that side right now, since
        rotating the whole side away drops both of its wheels' live readings together, not
        just one.

        Non-paired tags (chassis, windshield, roof) have no "side" to check against, so they
        used to have no protection at all here - and chassis specifically paid for it (Issue
        #18, 2026-09-11): attaching a wheel means moving the car around, occasionally blocking
        the classifier's view of the chassis with a hand for a while, which is enough sustained
        low confidence to un-confirm chassis despite the car never leaving the mat. Since
        _furthest_completed() requires chassis for every single step, losing it mid-assembly
        collapsed guidance all the way back to step 1 and forced re-climbing the whole
        sequence, which is exactly the "asking for chassis again and again" symptom. These tags
        now require the depth-based physical-presence signal (Issue #11) to *also* show
        sustained absence before un-confirming - genuinely empty mat, not just classifier
        confusion while something (the car, a hand) is still right there.
        """
        if len(self._recent_long[tag]) < self.unconfirm_window_frames:
            return
        true_fraction = sum(self._recent_long[tag]) / self.unconfirm_window_frames
        if true_fraction > self.unconfirm_max_fraction:
            return

        side = self._tag_side.get(tag)
        if side is not None:
            if side not in sides_in_view:
                return  # side isn't currently in view - could just be rotated away
        else:
            if len(self._recent_presence) < self.unconfirm_window_frames:
                return
            presence_fraction = sum(self._recent_presence) / self.unconfirm_window_frames
            if presence_fraction > self.unconfirm_max_fraction:
                return  # something's still physically on the mat - not genuine removal

        self.session_state[tag] = False
        self._recent[tag].clear()  # start the reconfirmation window fresh

    def _furthest_completed(self):
        # states.json is cumulative and ordered, so the first unmet step
        # marks the end of what's been completed - no need to scan further.
        completed = None
        for state in self.states:
            if all(self.session_state.get(t, False) for t in state["requires"]):
                completed = state
            else:
                break
        return completed

    def _next_after(self, completed):
        if completed is None:
            return self.states[0] if self.states else None
        idx = self.states.index(completed)
        if idx + 1 < len(self.states):
            return self.states[idx + 1]
        return None  # every step complete

    def _sides_in_view(self, raw_true):
        return {
            side for side, side_tags in self.side_groups.items()
            if any(raw_true.get(t, False) for t in side_tags)
        }

    def update(self, tag_probs: Dict[str, float], physical_presence: Optional[bool] = None) -> Guidance:
        """Feed one frame's per-tag probabilities, get back the current guidance.

        physical_presence, when provided, is a depth-sensor-derived reading of
        whether anything is physically elevated on the mat right now -
        completely independent of the RGB classifier, so it can't share its
        failure modes. False forces every tag's raw reading false for this
        frame regardless of classifier confidence (Issue #11, 2026-09-11):
        inference_loop.py already skips classifying a frame entirely when
        the depth presence check fails, so _recent never grows on those
        frames - but that's frame-count-based, not time-based, so a rare
        false-positive presence blip (sensor noise, a shadow) combined with
        the classifier being confidently wrong during that same instant
        could still slowly accumulate toward confirmation over a long
        enough session. This is the same signal checked again, directly in
        the confirmation path, as a second independent layer rather than
        relying solely on the caller skipping the frame upstream.
        """
        # None (presence unknown/not wired up by the caller) is treated as
        # "presumed present" - _maybe_unconfirm's non-paired-tag check should
        # never become the active blocker when the caller isn't supplying
        # this signal at all, only when it's actively confirming absence.
        self._recent_presence.append(True if physical_presence is None else physical_presence)

        raw_true = {
            t: tag_probs.get(t, 0.0) > config.TAG_CONFIDENCE_THRESHOLDS.get(t, self.threshold)
            for t in self.tags
        }
        if physical_presence is False:
            raw_true = {t: False for t in self.tags}

            # Auto-reset on a sustained, depth-confirmed empty mat (Issue
            # #12, 2026-09-11) - one-way confirmation means nothing else
            # clears stale confirmations between repeated test runs, so
            # without this, every retest inherits whatever the previous
            # attempt wrongly locked in. Only runs while there's something
            # to actually clear, and needs many consecutive empty frames
            # (not just one) so a hand briefly passing over the mat can't
            # trigger it.
            if any(self.session_state.values()):
                self._empty_streak += 1
                if self._empty_streak >= self.empty_reset_frames:
                    self.reset()  # raw_true is already all-False for this frame either way
        else:
            self._empty_streak = 0

        sides_in_view = self._sides_in_view(raw_true)

        for t in self.tags:
            if self.session_state[t]:
                self._recent_long[t].append(raw_true[t])
                self._maybe_unconfirm(t, sides_in_view)
            else:
                self._recent[t].append(raw_true[t])

        # Non-paired tags (chassis, windshield, roof) confirm independently,
        # same as always: sustained majority evidence over their own window.
        for t in self.tags:
            if t in self._paired_tags or self.session_state[t]:
                continue
            if len(self._recent[t]) == self.window_frames:
                true_fraction = sum(self._recent[t]) / self.window_frames
                if true_fraction >= self.min_true_fraction:
                    self.session_state[t] = True
                    self._recent_long[t].clear()  # start the removal-evidence window fresh

        # Paired (wheel) tags confirm together, not independently (Issue #10,
        # 2026-09-11): a side only confirms once BOTH its tags have
        # simultaneously satisfied the same window/fraction bar on their own
        # evidence. Without this, a spurious sustained reading on just one
        # wheel could permanently confirm a whole side that was never
        # attached - entry needs to be harder to fake without making it
        # slower for a genuine attachment (both wheels of a real side
        # naturally cross their windows around the same time anyway, so
        # this costs no latency).
        for side, side_tags in self.side_groups.items():
            if all(self.session_state[t] for t in side_tags):
                continue  # already confirmed
            ready = all(
                len(self._recent[t]) == self.window_frames
                and (sum(self._recent[t]) / self.window_frames) >= self.min_true_fraction
                for t in side_tags
            )
            if ready:
                for t in side_tags:
                    self.session_state[t] = True
                    self._recent_long[t].clear()

        completed = self._furthest_completed()
        next_step = self._next_after(completed)

        mismatch = False
        message = None
        instructions = None
        for side, side_tags in self.side_groups.items():
            if side not in sides_in_view:
                continue
            confirmed = [t for t in side_tags if self.session_state[t]]
            missing = [t for t in side_tags if not self.session_state[t]]
            if confirmed and missing:
                # side_groups pairs exactly 2 tags per side (front + rear), so
                # this branch only fires with exactly one of the two missing -
                # naming which specific wheel (front vs rear) isn't useful to
                # a trainee here, just that side's wheel isn't fully attached.
                mismatch = True
                message = f"{side.capitalize()} wheel isn't connected."
                break

        if message is None:
            if next_step is None:
                message = "Assembly complete!"
            else:
                # states.json is cumulative/ordered, so this index doubles as
                # a progress counter - it jumps forward on its own whenever
                # several steps get confirmed at once (assembly ahead of
                # guidance), since next_step always follows the furthest
                # step whose requirements are actually satisfied right now.
                step_num = self.states.index(next_step) + 1
                message = f"Step {step_num}/{len(self.states)}: {next_step['label']}"
                instructions = next_step.get("instructions")

        return Guidance(
            session_state=dict(self.session_state),
            completed_step=completed["step"] if completed else None,
            next_step=next_step,
            message=message,
            instructions=instructions,
            mismatch=mismatch,
        )
