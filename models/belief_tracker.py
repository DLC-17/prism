"""3-Segment Deck & Bayesian Belief Tracker for prism."""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional


class DeckBeliefTracker:
    """Maintains the 3-segment deck belief tensor T_deck_belief in R^{B x 3 x CardPoolSize}

    - Segment 1 (Top Stack): L_top <= 3 slots, deterministic discrete card IDs.
    - Segment 2 (Middle Pool): Hypergeometric count probability distribution.
    - Segment 3 (Bottom Queue): Ordered list L_bot <= 10 slots (Iono/bottom-deck).
    """

    def __init__(self, card_pool_size: int = 2048):
        self.card_pool_size = card_pool_size
        self.top_stack: deque[int] = deque()
        self.middle_pool: Counter[int] = Counter()
        self.bottom_queue: deque[int] = deque()

    def reset_with_deck(self, deck_cards: List[int]):
        self.top_stack.clear()
        self.middle_pool = Counter(deck_cards)
        self.bottom_queue.clear()

    def trigger_shuffle(self):
        """EVENT_SHUFFLE: Merges Segment 1 and Segment 3 into Segment 2.

        All positional certitude is reset to uniform multinomial sampling.
        """
        while self.top_stack:
            c = self.top_stack.popleft()
            self.middle_pool[c] += 1
        while self.bottom_queue:
            c = self.bottom_queue.popleft()
            self.middle_pool[c] += 1

    def trigger_ciphermaniac(self, card_a: int, card_b: int):
        """EVENT_CIPHERMANIAC: Inserts 2 chosen card IDs into Segment 1 at index 0 and 1."""
        for c in (card_a, card_b):
            if self.middle_pool[c] > 0:
                self.middle_pool[c] -= 1
        self.top_stack.appendleft(card_b)
        self.top_stack.appendleft(card_a)

    def trigger_iono(self, bottom_cards: List[int]):
        """EVENT_IONO: Hand cards moved to Segment 3 in randomized order."""
        for c in bottom_cards:
            if len(self.bottom_queue) >= 10:
                displaced = self.bottom_queue.popleft()
                self.middle_pool[displaced] += 1
            self.bottom_queue.append(c)

    def to_tensor(self) -> torch.Tensor:
        """Constructs T_deck_belief in R^{3 x CardPoolSize}."""
        t = torch.zeros(3, self.card_pool_size, dtype=torch.float32)

        # Segment 1: Top Stack
        if self.top_stack:
            for i, c in enumerate(self.top_stack):
                if c < self.card_pool_size:
                    t[0, c] += 1.0 / len(self.top_stack)

        # Segment 2: Middle Pool Hypergeometric count distribution
        total_mid = sum(self.middle_pool.values())
        if total_mid > 0:
            for c, cnt in self.middle_pool.items():
                if c < self.card_pool_size and cnt > 0:
                    t[1, c] = float(cnt) / float(total_mid)

        # Segment 3: Bottom Queue
        if self.bottom_queue:
            for c in self.bottom_queue:
                if c < self.card_pool_size:
                    t[2, c] += 1.0 / len(self.bottom_queue)

        return t

    @staticmethod
    def solve_exact_prizes(
        initial_60: List[int],
        visible_cards: List[int],
    ) -> List[int]:
        """Deterministic Deck-Search Resolution:

        Prize_exact = InitialDeck_60 - (VisibleBoard + Hand + Discard + RemainingDeck)
        Prize tokens transition instantly from uniform belief to exact card IDs.
        """
        counts = Counter(initial_60)
        for c in visible_cards:
            if counts[c] > 0:
                counts[c] -= 1

        exact_prizes = []
        for c, cnt in counts.items():
            exact_prizes.extend([c] * cnt)
        return exact_prizes


class BayesianArchetypeTracker:
    """Maintains a 32-dimensional categorical posterior over competitive meta archetypes

    updated via Bayes' rule on revealed cards.
    """

    def __init__(self, num_archetypes: int = 32, card_pool_size: int = 2048):
        self.num_archetypes = num_archetypes
        self.card_pool_size = card_pool_size

        # Uniform prior
        self.log_prior = np.zeros(num_archetypes, dtype=np.float32)

        # Likelihood table: P(card | archetype)
        # Seeded with characteristic signature cards
        self.log_likelihood = np.full((num_archetypes, card_pool_size), -6.0, dtype=np.float32)
        self._initialize_signatures()

        self.current_log_posterior = self.log_prior.copy()

    def _initialize_signatures(self):
        # Archetype 0: Charizard ex / Pidgeot ex (Signature cards: Charizard ex id 12, Charmander id 10, Rare Candy id 15)
        self.log_likelihood[0, 10] = -0.1
        self.log_likelihood[0, 11] = -0.5
        self.log_likelihood[0, 12] = -0.05

        # Archetype 1: Lugia VSTAR (Signature cards: Archeops id 20, Lugia id 21)
        self.log_likelihood[1, 20] = -0.05
        self.log_likelihood[1, 21] = -0.05

        # Archetype 2: Gardevoir ex (Signature cards: Ralts id 30, Kirlia id 31, Gardevoir ex id 32)
        self.log_likelihood[2, 30] = -0.05
        self.log_likelihood[2, 31] = -0.05
        self.log_likelihood[2, 32] = -0.05

        # Archetype 3: Raging Bolt ex / Ogerpon (Signature cards: Raging Bolt id 40, Teal Mask id 41)
        self.log_likelihood[3, 40] = -0.05
        self.log_likelihood[3, 41] = -0.05

        # Archetype 4: Hydrapple ex / Teal Mask Ogerpon ex (Signature cards: Applin id 201, Dipplin id 202, Hydrapple ex id 203, Ogerpon id 204, Bug Catching id 205)
        self.log_likelihood[4, 201] = -0.05
        self.log_likelihood[4, 202] = -0.1
        self.log_likelihood[4, 203] = -0.01
        self.log_likelihood[4, 204] = -0.05
        self.log_likelihood[4, 205] = -0.1

    def observe_card(self, card_id: int):
        """Bayesian update on observing an opponent card."""
        if card_id < self.card_pool_size:
            self.current_log_posterior += self.log_likelihood[:, card_id]
            # Numerical stabilization: subtract max
            self.current_log_posterior -= np.max(self.current_log_posterior)

    def get_posterior(self) -> np.ndarray:
        exp_post = np.exp(self.current_log_posterior)
        s = np.sum(exp_post)
        if s > 0:
            return (exp_post / s).astype(np.float32)
        return np.full(self.num_archetypes, 1.0 / self.num_archetypes, dtype=np.float32)


class MatchContextBuilder:
    """Builds the 64-dimensional match context vector z_match:

    - opponent_archetype_posterior: R^{32}
    - series_score: R^{3} (Game 1, Leading 1-0, Trailing 0-1, Tied 1-1 / wins)
    - match_clock_remaining: R^{1} (0.0 to 1.0, 50-minute tournament clock)
    - auxiliary: R^{28} (prizes taken self/opp, turn number, player order)
    """

    @staticmethod
    def build(
        archetype_posterior: np.ndarray,  # (32,)
        series_score: List[float],        # (3,)
        match_clock_remaining: float,    # scalar
        self_prizes_taken: int = 0,
        opp_prizes_taken: int = 0,
        turn_count: int = 1,
        is_first_player: bool = True,
    ) -> torch.Tensor:
        z = np.zeros(64, dtype=np.float32)

        # Archetype posterior (32)
        z[:32] = archetype_posterior[:32]

        # Series score (3)
        z[32:35] = series_score[:3]

        # Clock remaining (1)
        z[35] = match_clock_remaining

        # Auxiliary features (28)
        z[36] = self_prizes_taken / 6.0
        z[37] = opp_prizes_taken / 6.0
        z[38] = turn_count / 30.0
        z[39] = 1.0 if is_first_player else 0.0

        return torch.from_numpy(z).unsqueeze(0)  # (1, 64)
