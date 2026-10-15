"""Devolution knockout check implementation in Python."""

from dataclasses import dataclass


@dataclass
class DevolutionCheckResult:
    is_devolvable: bool
    under_slot_1_hp: int
    current_damage: int
    is_devol_ko: bool


def check_devolution_ko(
    damage_counters: int,
    evolution_depth: int,
    under_slot_1_hp: int,
) -> DevolutionCheckResult:
    """Evaluates devolution knockout according to the architectural specification:

    UnderHP = e_dynamic[under_slot_1_hp]
    IsDevolvable = I(evolution_depth >= 1)
    DevolKO = IsDevolvable and (damage_counters * 10 >= UnderHP)
    """
    is_devolvable = evolution_depth >= 1
    current_damage = damage_counters * 10
    is_devol_ko = is_devolvable and (current_damage >= under_slot_1_hp)

    return DevolutionCheckResult(
        is_devolvable=is_devolvable,
        under_slot_1_hp=under_slot_1_hp,
        current_damage=current_damage,
        is_devol_ko=is_devol_ko,
    )
