"""Registers the Hydrapple ex / Teal Mask Ogerpon ex deck cards into the card catalog."""

from engine.python_fallback.card_db import CardCatalog, CardDef, AttackDef
from engine.python_fallback.types import CardCategory, EnergyType


def register_hydrapple_archetype(catalog: CardCatalog):
    """Registers the #1 rated Kaggle playground archetype:

    Hydrapple ex / Teal Mask Ogerpon ex / Dipplin (Grass acceleration).
    """
    # 200: Basic Grass Energy
    catalog.insert(
        CardDef(
            card_id=200,
            name="Basic Grass Energy",
            category=CardCategory.BASIC_ENERGY,
            stage=0,
            base_hp=0,
            primary_type=EnergyType.GRASS,
        )
    )

    # 201: Applin (Root basic)
    catalog.insert(
        CardDef(
            card_id=201,
            name="Applin",
            category=CardCategory.BASIC_POKEMON,
            stage=0,
            base_hp=40,
            primary_type=EnergyType.GRASS,
            weakness=EnergyType.FIRE,
            retreat_cost=1,
            attacks=[
                AttackDef(
                    name="Spray Fluid",
                    energy_cost=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    base_damage=10,
                )
            ],
        )
    )

    # 202: Dipplin (Stage 1)
    catalog.insert(
        CardDef(
            card_id=202,
            name="Dipplin",
            category=CardCategory.STAGE_1_POKEMON,
            stage=1,
            base_hp=80,
            primary_type=EnergyType.GRASS,
            weakness=EnergyType.FIRE,
            retreat_cost=2,
            attacks=[
                AttackDef(
                    name="Do the Wave",
                    energy_cost=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    base_damage=20,
                )
            ],
            evolves_from_id=201,
        )
    )

    # 203: Hydrapple ex (Stage 2 ex) - #1 Meta Engine
    # Ripening Charge (heal 30 + attach Grass energy) + Syrup Storm (30 + 30 per grass energy)
    catalog.insert(
        CardDef(
            card_id=203,
            name="Hydrapple ex",
            category=CardCategory.EX_POKEMON,
            stage=4,  # ex stage
            base_hp=330,
            primary_type=EnergyType.GRASS,
            weakness=EnergyType.FIRE,
            retreat_cost=3,
            attacks=[
                AttackDef(
                    name="Syrup Storm",
                    energy_cost=[0, 0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2 Colorless
                    base_damage=30,
                    effect_flag_id=30,  # +30 per grass energy
                )
            ],
            ability_category=2,  # Energy attachment + Heal
            evolves_from_id=202,
        )
    )

    # 204: Teal Mask Ogerpon ex (Basic ex)
    # Teal Dance (attach Grass energy + draw 1) + Myriad Leaf Shower
    catalog.insert(
        CardDef(
            card_id=204,
            name="Teal Mask Ogerpon ex",
            category=CardCategory.EX_POKEMON,
            stage=4,  # Basic ex
            base_hp=210,
            primary_type=EnergyType.GRASS,
            weakness=EnergyType.FIRE,
            retreat_cost=1,
            attacks=[
                AttackDef(
                    name="Myriad Leaf Shower",
                    energy_cost=[3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 Grass
                    base_damage=20,
                    effect_flag_id=31,  # +30 per attached energy
                )
            ],
            ability_category=1,  # Draw + Energy attach
        )
    )

    # 205: Bug Catching Set (Item)
    catalog.insert(
        CardDef(
            card_id=205,
            name="Bug Catching Set",
            category=CardCategory.ITEM,
            stage=0,
            base_hp=0,
        )
    )


def create_hydrapple_deck_list() -> list[int]:
    """Generates the standard competitive 60-card Hydrapple / Ogerpon decklist."""
    deck = []
    deck.extend([201] * 4)  # 4 Applin
    deck.extend([202] * 3)  # 3 Dipplin
    deck.extend([203] * 3)  # 3 Hydrapple ex
    deck.extend([204] * 3)  # 3 Teal Mask Ogerpon ex
    deck.extend([205] * 4)  # 4 Bug Catching Set
    deck.extend([1] * 4)    # 4 Ultra Ball
    deck.extend([200] * 14) # 14 Basic Grass Energy
    deck.extend([100] * 5)  # 5 Other items/supporters
    # Pad to 60 cards
    while len(deck) < 60:
        deck.append(200)
    return deck[:60]
