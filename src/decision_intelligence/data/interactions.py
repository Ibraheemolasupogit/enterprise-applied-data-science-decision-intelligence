"""Sessions, recommendation exposures and engagement events.

Two latent mechanisms drive this module, deliberately kept apart:

* **True preference** — a per-customer affinity over product categories
  (``latent_affinities``), hidden from any naive observer and used only to
  decide whether a shown item gets clicked/purchased.
* **Exposure mechanism** — a mix of ``"personalized"`` (affinity-weighted)
  and ``"popularity"`` (Zipf-weighted, affinity-blind) item selection for
  what actually gets *shown*.

Because popularity-driven exposure ignores affinity, popular items are
over-shown relative to how well they actually fit each customer — the
exposure bias that offline recommender evaluation (M8/M9) must account for
rather than mistake for true preference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_intelligence.data.config import SyntheticDataConfig
from decision_intelligence.data.customers import horizon_end
from decision_intelligence.data.products import CATEGORIES

_PERSONALIZED_EXPOSURE_SHARE = 0.7
_BASE_CLICK_RATE = 0.35
_PURCHASE_GIVEN_CLICK_RATE = 0.25
_AFFINITY_CONCENTRATION = 2.0


def generate_latent_affinities(
    config: SyntheticDataConfig, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw each customer's hidden affinity over product categories.

    Ground truth only: this table must never be used as a model feature —
    it exists so recommender evaluation (M8/M9) has a known "right answer"
    to measure ranking quality against.
    """
    n = len(customers)
    affinities = rng.dirichlet(np.full(len(CATEGORIES), _AFFINITY_CONCENTRATION), size=n)
    frame = pd.DataFrame(affinities, columns=list(CATEGORIES))
    frame.insert(0, "customer_id", customers["customer_id"].to_numpy())
    return frame


def _popularity_weights(products: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    # Zipf-like popularity: a short head of very popular items, long tail of
    # rarely-shown ones, independent of category or true affinity.
    rank = rng.permutation(len(products)) + 1
    weights = 1.0 / rank
    return np.asarray(weights / weights.sum())


def generate_interactions(
    config: SyntheticDataConfig,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    subscriptions: pd.DataFrame,
    latent_affinities: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sessions, recommendation exposures and click/purchase events."""
    popularity_weights = _popularity_weights(products, rng)
    product_category = products["category"].to_numpy()
    product_ids = products["product_id"].to_numpy()

    affinity_by_customer = latent_affinities.set_index("customer_id")
    subscription_window = subscriptions.set_index("customer_id")[["start_date", "end_date"]]

    session_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    session_id_counter = 1
    exposure_id_counter = 1
    event_id_counter = 1

    for customer in customers.itertuples(index=False):
        customer_id = int(customer.customer_id)  # type: ignore[arg-type]
        window_end_value = subscription_window.loc[customer_id, "end_date"]
        window_end = (
            pd.Timestamp(window_end_value)  # type: ignore[arg-type]
            if pd.notna(window_end_value)  # type: ignore[arg-type]
            else horizon_end(config)
        )
        window_start = pd.Timestamp(
            subscription_window.loc[customer_id, "start_date"]  # type: ignore[arg-type]
        )
        active_days = max(int((window_end - window_start).days), 0)
        if active_days == 0:
            continue

        n_sessions = rng.poisson(config.sessions_per_customer_mean)
        if n_sessions == 0:
            continue

        session_offsets = rng.integers(0, active_days, size=n_sessions)
        session_timestamps = window_start + pd.to_timedelta(session_offsets, unit="D")

        customer_affinity = affinity_by_customer.loc[customer_id, list(CATEGORIES)].to_numpy(
            dtype=float
        )
        item_affinity = customer_affinity[[CATEGORIES.index(c) for c in product_category]]
        personalized_weights = item_affinity / item_affinity.sum()

        for session_timestamp in session_timestamps:
            session_id = session_id_counter
            session_id_counter += 1
            session_rows.append(
                {
                    "session_id": session_id,
                    "customer_id": customer_id,
                    "session_timestamp": session_timestamp,
                }
            )

            n_items = max(int(rng.poisson(config.items_per_session_mean)), 1)
            use_personalized = rng.random(n_items) < _PERSONALIZED_EXPOSURE_SHARE
            chosen_index = np.empty(n_items, dtype=int)
            n_personalized = int(use_personalized.sum())
            if n_personalized:
                chosen_index[use_personalized] = rng.choice(
                    len(product_ids), size=n_personalized, p=personalized_weights
                )
            n_popularity = n_items - n_personalized
            if n_popularity:
                chosen_index[~use_personalized] = rng.choice(
                    len(product_ids), size=n_popularity, p=popularity_weights
                )

            for position, (item_index, personalized) in enumerate(
                zip(chosen_index, use_personalized, strict=False), start=1
            ):
                exposure_id = exposure_id_counter
                exposure_id_counter += 1
                product_id = int(product_ids[item_index])
                exposure_rows.append(
                    {
                        "exposure_id": exposure_id,
                        "session_id": session_id,
                        "customer_id": customer_id,
                        "product_id": product_id,
                        "position": position,
                        "exposure_mechanism": "personalized" if personalized else "popularity",
                    }
                )

                true_affinity = item_affinity[item_index]
                click_probability = min(_BASE_CLICK_RATE * true_affinity * len(CATEGORIES), 0.95)
                clicked = rng.random() < click_probability
                if not clicked:
                    continue
                event_rows.append(
                    {
                        "event_id": event_id_counter,
                        "exposure_id": exposure_id,
                        "event_type": "click",
                        "event_timestamp": session_timestamp,
                        "recorded_at": session_timestamp,
                    }
                )
                event_id_counter += 1

                if rng.random() < _PURCHASE_GIVEN_CLICK_RATE:
                    event_rows.append(
                        {
                            "event_id": event_id_counter,
                            "exposure_id": exposure_id,
                            "event_type": "purchase",
                            "event_timestamp": session_timestamp,
                            "recorded_at": session_timestamp,
                        }
                    )
                    event_id_counter += 1

    sessions = pd.DataFrame(session_rows)
    exposures = pd.DataFrame(exposure_rows)
    events = pd.DataFrame(event_rows)
    return sessions, exposures, events
