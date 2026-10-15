"""Tests for Devolution Knockout Check logic."""

from engine.python_fallback.devo import check_devolution_ko


def test_basic_pokemon_devolution():
    # Basic Pokemon cannot be devolved
    res = check_devolution_ko(damage_counters=10, evolution_depth=0, under_slot_1_hp=0)
    assert res.is_devolvable is False
    assert res.is_devol_ko is False


def test_stage_1_surviving_devolution():
    # Drakloak (Stage 1) with 4 damage counters (40 damage).
    # Dreepy (under_slot_1) has 60 HP.
    # 40 < 60 -> Survives!
    res = check_devolution_ko(damage_counters=4, evolution_depth=1, under_slot_1_hp=60)
    assert res.is_devolvable is True
    assert res.current_damage == 40
    assert res.is_devol_ko is False


def test_stage_1_knockout_devolution():
    # Drakloak (Stage 1) with 6 damage counters (60 damage).
    # Dreepy (under_slot_1) has 60 HP.
    # 60 >= 60 -> Devolution KO!
    res = check_devolution_ko(damage_counters=6, evolution_depth=1, under_slot_1_hp=60)
    assert res.is_devolvable is True
    assert res.current_damage == 60
    assert res.is_devol_ko is True


def test_stage_2_rare_candy_devolution():
    # Charizard ex evolved via Rare Candy (evolution_depth = 2, under_slot_1 is Charmander 70 HP)
    # With 7 damage counters (70 damage) -> 70 >= 70 -> KO!
    res = check_devolution_ko(damage_counters=7, evolution_depth=2, under_slot_1_hp=70)
    assert res.is_devolvable is True
    assert res.current_damage == 70
    assert res.is_devol_ko is True
