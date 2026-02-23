use std::{collections::HashMap, sync::LazyLock};

use shakmaty::uci::UciMove;

const ALL_MOVES_JSON: &str = include_str!("data/all_moves.json");

pub static ALL_MOVES: LazyLock<HashMap<UciMove, usize>> = LazyLock::new(|| {
    let parsed: HashMap<String, usize> =
        serde_json::from_str(ALL_MOVES_JSON).expect("Failed to parse all_moves.json");

    parsed
        .into_iter()
        .map(|(uci_str, idx)| {
            let uci_move: UciMove = uci_str.parse().unwrap();
            (uci_move, idx)
        })
        .collect()
});

// const ALL_MOVES_REVERSED_JSON: &str = include_str!("data/all_moves_reversed.json");
// pub static ALL_MOVES_REVERSED: LazyLock<HashMap<usize, String>> = LazyLock::new(|| {
//     serde_json::from_str(ALL_MOVES_REVERSED_JSON).expect("Failed to parse all_moves_reversed.json")
// });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_moves_loaded() {
        dbg!(&*ALL_MOVES);
        assert!(!ALL_MOVES.is_empty());
    }
}
