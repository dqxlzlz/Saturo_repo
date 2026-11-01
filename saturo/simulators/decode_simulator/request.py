"""Request data models for the decode simulator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class RequestType(Enum):
    RT = "RT"
    BE = "BE"


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Request:
    """A single inference request."""
    request_id: int
    request_type: RequestType
    prompt_len: int
    target_output_len: int
    arrival_time_s: float

    # Runtime state
    state: RequestState = RequestState.WAITING
    tokens_generated: int = 0
    prefill_start_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    finish_time: Optional[float] = None

    # For BE chunked prefill
    prefill_tokens_processed: int = 0

    # Token IDs (optional, populated when loaded from WildChat parquet).
    # prompt_token_ids: the actual prompt token sequence.
    # output_token_ids: the full ground-truth output sequence; during simulation
    #   output_token_ids[:tokens_generated] gives the generated prefix so far.
    # Both are None when loaded from Azure CSV traces (length-only).
    prompt_token_ids: Optional[Tuple[int, ...]] = field(default=None, repr=False)
    output_token_ids: Optional[Tuple[int, ...]] = field(default=None, repr=False)

    @property
    def is_active_decode(self) -> bool:
        return self.state == RequestState.DECODING

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    @property
    def current_seq_len(self) -> int:
        return self.prompt_len + self.tokens_generated

    @property
    def remaining_output(self) -> int:
        return max(self.target_output_len - self.tokens_generated, 0)

    @property
    def remaining_prefill(self) -> int:
        return max(self.prompt_len - self.prefill_tokens_processed, 0)

    @property
    def generated_token_ids(self) -> Optional[Tuple[int, ...]]:
        """The generated token prefix so far (slice of output_token_ids).

        Returns None if output_token_ids is not available (e.g. Azure CSV trace).
        """
        if self.output_token_ids is None:
            return None
        return self.output_token_ids[:self.tokens_generated]

    def generate_one_token(self):
        self.tokens_generated += 1
        if self.tokens_generated >= self.target_output_len:
            self.state = RequestState.FINISHED

    def process_prefill_chunk(self, chunk_size: int) -> int:
        """Process a chunk of prefill tokens. Returns actual tokens processed."""
        actual = min(chunk_size, self.remaining_prefill)
        self.prefill_tokens_processed += actual
        if self.prefill_tokens_processed >= self.prompt_len:
            self.state = RequestState.DECODING
        return actual
