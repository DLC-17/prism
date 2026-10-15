"""Pure-Python simulation mirror for the prism Pokémon TCG rule engine."""

from .types import (
    ZoneType,
    Zone,
    EnergyType,
    CardCategory,
    MacroActionType,
    SubState,
    D_MODEL,
    N_ENTITIES,
    CARD_POOL_SIZE,
    RAW_FEATURE_DIM,
)
from .card_db import CardDef, AttackDef, CardCatalog
from .zone_manager import PokemonEntity, ThreeSegmentDeck, encode_zone
from .state_machine import GameState
from .devo import check_devolution_ko, DevolutionCheckResult

__all__ = [
    "ZoneType",
    "Zone",
    "EnergyType",
    "CardCategory",
    "MacroActionType",
    "SubState",
    "D_MODEL",
    "N_ENTITIES",
    "CARD_POOL_SIZE",
    "RAW_FEATURE_DIM",
    "CardDef",
    "AttackDef",
    "CardCatalog",
    "PokemonEntity",
    "ThreeSegmentDeck",
    "encode_zone",
    "GameState",
    "check_devolution_ko",
    "DevolutionCheckResult",
]
