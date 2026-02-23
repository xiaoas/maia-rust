use std::{collections::HashMap, sync::LazyLock};

use shakmaty::uci::UciMove;

// JSON representation of the fixed move vocabulary used by Maia2.  The
// file maps UCI strings to indices in the model's output layer.  The
// `include_str!` macro embeds the data at compile time, avoiding an
// extra filesystem dependency at runtime.
const ALL_MOVES_JSON: &str = include_str!("data/all_moves.json");

/// Mapping from `UciMove` to the corresponding output index.
///
/// This map is lazily initialized on first access.  Using
/// `LazyLock` avoids the cost of parsing the JSON when the crate is
/// simply linked but not used.
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

// The reversed mapping (index -> UCI) is not currently used but is
// kept here in comments in case future features need to convert the
// model's integer outputs back to strings.
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
