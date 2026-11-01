"""Abstract interface every model adapter must implement.

A model adapter encapsulates everything *model-specific*:
  - how to load weights,
  - how to decode the video (12 fixed frames? fps=2.0?),
  - how to ask QA, and
  - how to ask for a temporal interval.

The protocol scripts (supportgain/protocol/) only talk to the adapter via this
interface, so adding a new model requires no changes to the pipeline.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np


class ModelAdapter(ABC):
    name = 'base'                 # override in subclass

    # ---- lifecycle -----------------------------------------------------

    @abstractmethod
    def load(self, device: str = 'cuda:0') -> None:
        """Load model weights and processor onto `device`."""

    # ---- video ---------------------------------------------------------

    @abstractmethod
    def decode_video(self, video_path: str
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decode the video the way THIS model natively consumes it.

        Returns:
            frames: [N, H, W, 3] uint8
            timestamps: [N] seconds
            duration: float
        """

    # ---- inference -----------------------------------------------------

    @abstractmethod
    def predict_qa(self, frames: np.ndarray, question: str,
                   choices: List[str]) -> Tuple[int, str]:
        """5-choice MCQ on the given frame stack.

        Returns:
            letter_index in 0..4 (-1 on parse fail)
            raw_text (for logging)
        """

    @abstractmethod
    def predict_grounding(self, frames: np.ndarray, question: str,
                          predicted_answer: str, duration: float,
                          video_path: Optional[str] = None,
                          ) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        """Temporal grounding for `question` given `predicted_answer`.

        `video_path` is optional — adapters that need to re-decode the
        video (e.g. HawkEye's recursive grounding samples frames within
        the narrowed bound at each turn) may use it. Adapters that do
        single-call grounding (e.g. Qwen2.5-VL) can ignore it.

        Returns:
            (start_sec, end_sec) or None on parse fail
            raw_text
            flag string (e.g. 'clamped', 'parse_fail') or None
        """

    # ---- optional hooks ------------------------------------------------

    def supports_blind_via_zero_frames(self) -> bool:
        """If True, the protocol uses zero-frames for Blind. Otherwise
        the adapter must implement `predict_blind` itself."""
        return True

    def predict_blind(self, frames: np.ndarray, question: str,
                      choices: List[str]) -> Tuple[int, str]:
        """Default Blind = run QA on all-zero frames."""
        return self.predict_qa(np.zeros_like(frames), question, choices)

    # ---- video-file-based intervention (opt-in) -----------------------
    #
    # Some models (e.g. MUPA) consume a video file rather than a frame
    # array and re-decode it internally with their own sampling rules.
    # For those models, encoding TEF's pre-decoded frames into a fresh
    # mp4 either loses information or breaks internal fps assumptions.
    #
    # An adapter that opts in by returning True from
    # ``supports_video_path_intervention`` will be called via
    # ``predict_qa_video`` instead of ``predict_qa`` from step1 and
    # step3. step3 supplies the intervention spec so the adapter can
    # apply it at the video-file level (e.g. via ffmpeg).
    #
    # ``intervention_spec`` is either None (full inference, no
    # modification) or a dict::
    #
    #     {
    #         'phase':  'phase1' | 'phase2' | 'phase3',
    #         'kind':   'keep' | 'remove' | 'blind' | 'full',
    #         'interval': (start_sec, end_sec) | None,
    #         'total_duration': float,
    #     }

    def supports_video_path_intervention(self) -> bool:
        return False

    def predict_qa_video(self, video_path: str,
                        intervention_spec: Optional[dict],
                        question: str, choices: List[str]
                        ) -> Tuple[int, str]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_qa_video"
        )
