"""Domain types, constants, and enums for the pure-Python simulation mirror."""

from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional

D_MODEL: int = 256
N_ENTITIES: int = 64
CARD_POOL_SIZE: int = 2048

E_STATIC_DIM: int = 128
E_DYNAMIC_DIM: int = 96
E_ZONE_DIM: int = 32
RAW_FEATURE_DIM: int = E_STATIC_DIM + E_DYNAMIC_DIM + E_ZONE_DIM  # 256


class ZoneType(IntEnum):
    ACTIVE_SELF = 0
    BENCH_SELF = 1
    HAND_SELF = 2
    DISCARD_SELF = 3
    LOST_SELF = 4
    PRIZE_SELF = 5
    ACTIVE_OPP = 6
    BENCH_OPP = 7
    HAND_OPP_REVEALED = 8
    DISCARD_OPP = 9
    LOST_OPP = 10
    STADIUM = 11


@dataclass(frozen=True)
class Zone:
    zone_type: Optional[ZoneType]
    slot_index: int = 0
    is_self: bool = True

    @property
    def controller(self) -> float:
        if self.zone_type in (
            ZoneType.ACTIVE_SELF,
            ZoneType.BENCH_SELF,
            ZoneType.HAND_SELF,
            ZoneType.DISCARD_SELF,
            ZoneType.LOST_SELF,
            ZoneType.PRIZE_SELF,
        ):
            return 1.0
        elif self.zone_type in (
            ZoneType.ACTIVE_OPP,
            ZoneType.BENCH_OPP,
            ZoneType.HAND_OPP_REVEALED,
            ZoneType.DISCARD_OPP,
            ZoneType.LOST_OPP,
        ):
            return -1.0
        return 0.0


class EnergyType(IntEnum):
    GRASS = 0
    FIRE = 1
    WATER = 2
    LIGHTNING = 3
    PSYCHIC = 4
    FIGHTING = 5
    DARKNESS = 6
    METAL = 7
    COLORLESS = 8
    SPECIAL = 9


class CardCategory(IntEnum):
    BASIC_POKEMON = 0
    STAGE_1_POKEMON = 1
    STAGE_2_POKEMON = 2
    VSTAR_POKEMON = 3
    EX_POKEMON = 4
    ITEM = 5
    SUPPORTER = 6
    TOOL = 7
    STADIUM = 8
    BASIC_ENERGY = 9
    SPECIAL_ENERGY = 10


class MacroActionType(IntEnum):
    ATTACK = 0
    RETREAT = 1
    PLAY_ITEM = 2
    PLAY_SUPPORTER = 3
    ATTACH_ENERGY = 4
    USE_ABILITY = 5
    PASS = 6


class SubState(Enum):
    IDLE_ROOT = "idle_root"
    SELECT_DISCARD_1 = "select_discard_1"
    SELECT_DISCARD_2 = "select_discard_2"
    SELECT_SEARCH_TARGET = "select_search_target"
    SELECT_BENCH_TARGET = "select_bench_target"
    SELECT_ENERGY_TARGET = "select_energy_target"
    SELECT_RETREAT_COST = "select_retreat_cost"
