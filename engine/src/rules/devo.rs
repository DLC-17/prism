use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevolutionCheckResult {
    pub is_devolvable: bool,
    pub under_slot_1_hp: u16,
    pub current_damage: u16,
    pub is_devol_ko: bool,
}

/// Evaluates devolution knockout according to the architectural specification:
/// UnderHP = e_dynamic[under_slot_1_hp]
/// IsDevolvable = I(e_dynamic[evolution_depth] >= 1)
/// DevolKO = IsDevolvable and (damage_counters * 10 >= UnderHP)
pub fn check_devolution_ko(
    damage_counters: u8,
    evolution_depth: u8,
    under_slot_1_hp: u16,
) -> DevolutionCheckResult {
    let is_devolvable = evolution_depth >= 1;
    let current_damage = (damage_counters as u16) * 10;
    let is_devol_ko = is_devolvable && (current_damage >= under_slot_1_hp);

    DevolutionCheckResult {
        is_devolvable,
        under_slot_1_hp,
        current_damage,
        is_devol_ko,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_not_devolvable() {
        let result = check_devolution_ko(5, 0, 0);
        assert!(!result.is_devolvable);
        assert!(!result.is_devol_ko);
    }

    #[test]
    fn test_stage1_survives_devolution() {
        // e.g. Drakloak with 50 damage, Dreepy has 60 HP -> Survives (50 < 60)
        let result = check_devolution_ko(5, 1, 60);
        assert!(result.is_devolvable);
        assert_eq!(result.current_damage, 50);
        assert!(!result.is_devol_ko);
    }

    #[test]
    fn test_stage1_ko_on_devolution() {
        // e.g. Charizard ex with 80 damage devolved to Charmeleon (90 HP) -> Survives
        // But devolved to Charmander (70 HP) when under_slot_1 is Charmander -> KO! (80 >= 70)
        let result = check_devolution_ko(8, 2, 70);
        assert!(result.is_devolvable);
        assert_eq!(result.current_damage, 80);
        assert!(result.is_devol_ko);
    }
}
