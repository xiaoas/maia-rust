use ndarray::{Array4, ArrayViewMut3, Axis};
use shakmaty::{CastlingMode, Chess, Role, Setup, Square};

use crate::error::MaiaError;

/// Data produced by the preprocessing step, ready for model consumption.
///
/// - `board_tensor` has shape `[B, 18, 8, 8]` where `B` is the batch
///   size.  The 18 channels encode piece locations, turn, castling
///   rights, and en passant information.
/// - `mirrored` tracks which positions were horizontally mirrored to
///   force the model's perspective to always be White.
/// - `chess_positions` holds the corresponding `Chess` objects after
///   any mirroring has been applied.  This is needed when converting
///   the model's output back into human-readable UCI moves.
pub struct PreprocessedData {
    pub board_tensor: Array4<f32>, // Shape: [B, 18, 8, 8]
    /// maia2 trains on positions from the perspective of white. Black positions are mirrored before being fed into the model.
    pub mirrored: Vec<bool>,
    /// potentially mirrored chess positions. 
    pub chess_positions: Vec<Chess>,
}

/// Convert raw ELO ratings to the discrete category indices used by
/// Maia2 (0..=10).
///
/// - Ratings below 1100 map to `0`.
/// - Ratings 1100–1999 are bucketed in 100‑point increments.
/// - Ratings 2000 and above map to `10`.
pub fn map_elos_to_categories(elo: &[u32]) -> Vec<i64> {
    elo.iter()
        .map(|&e| {
            if e < 1100 {
                0
            } else if e >= 2000 {
                10
            } else {
                ((e - 1100) / 100 + 1) as i64
            }
        })
        .collect()
}

/// Transform an iterator of `Setup`s into the input tensors
/// expected by the Maia2 model.
///
/// `batch_size` must match the number of setups provided; mismatches
/// will panic.  This function also records whether each position was
/// mirrored and returns the possibly‑mirrored `Chess` objects for
/// later use.
pub fn preprocess(
    setups: impl IntoIterator<Item = Setup>,
    batch_size: usize,
) -> Result<PreprocessedData, MaiaError> {
    let mut board_tensor = Array4::<f32>::zeros((batch_size, 18, 8, 8));
    let mut mirrored_vec = Vec::with_capacity(batch_size);
    let mut chess_positions = Vec::with_capacity(batch_size);
    let mut last_index = 0;
    for (i, mut setup) in setups.into_iter().enumerate() {
        last_index = i;
        if i >= batch_size {
            panic!("More setups provided than batch size");
        }
        // If it's Black's turn we flip the board so the network always
        // sees White to move.  We remember this so that outputs can be
        // mirrored back later.
        let mirrored = setup.turn.is_black();
        mirrored_vec.push(mirrored);

        if mirrored {
            setup.mirror();
        }
        board_to_tensor(&setup, board_tensor.index_axis_mut(Axis(0), i));
        let position: Chess = setup.position(CastlingMode::Standard)?;
        chess_positions.push(position);
    }
    if last_index + 1 != batch_size {
        panic!("Fewer setups provided than batch size");
    }
    // let legal_move_indices = board.legal_moves()
    //     .into_iter()
    //     .map(|m| {
    //         let uci = m.to_uci(CastlingMode::Standard);
    //         *ALL_MOVES.get(&uci).unwrap() // All legal moves should be in the ALL_MOVES mapping
    //     })
    //     .collect();

    Ok(PreprocessedData {
        board_tensor,
        chess_positions,
        mirrored: mirrored_vec,
    })
}

fn board_to_tensor(setup: &Setup, mut tensor: ArrayViewMut3<f32>) {
    // 1. Piece placement occupies channels 0..11.  We encode white pieces
    // in 0..5 and black pieces in 6..11.
    for sq in Square::ALL {
        if let Some(piece) = setup.board.piece_at(sq) {
            let color_offset = if piece.color.is_white() { 0 } else { 6 };
            let role_offset = match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };
            let channel = color_offset + role_offset;
            // Native index conversion to row/file mapping (Rank 1 -> 0, File A -> 0)
            tensor[[channel, sq.rank() as usize, sq.file() as usize]] = 1.0;
        }
    }

    // 2. Player's turn (channel 12): set all squares to 1 if white to move.
    tensor
        .index_axis_mut(Axis(0), 12)
        .fill(setup.turn.is_white() as u8 as f32); // 1.0 if is_white else 0.0

    // 3. Castling rights (channels 13..16) indicate the original rook
    // squares; presence signals permission to castle.
    let castling_rights = [
        setup.castling_rights.contains(Square::H1), // K
        setup.castling_rights.contains(Square::A1), // Q
        setup.castling_rights.contains(Square::H8), // k
        setup.castling_rights.contains(Square::A8), // q
    ];
    for (i, &has_right) in castling_rights.iter().enumerate() {
        tensor
            .index_axis_mut(Axis(0), 13 + i)
            .fill(has_right as u8 as f32); // 1.0 if has_right else 0.0
    }

    // 4. En passant target (channel 17) is a one‑hot square if present.
    if let Some(ep_sq) = setup.ep_square {
        tensor[[17, ep_sq.rank() as usize, ep_sq.file() as usize]] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;

    #[test]
    fn verify_tensor_output() {
        // Position: Start of Sicilian Defense (1. e4 c5)
        let fen_str = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
        let setup: Setup = fen_str.parse::<Fen>().unwrap().into_setup();

        // Process a single item
        let data = preprocess(vec![setup], 1).unwrap();
        let tensor = data.board_tensor;

        println!("Rust Tensor Non-Zero Indices:");
        // Iterate over the single batch item (index 0)
        for c in 0..18 {
            for h in 0..8 {
                for w in 0..8 {
                    // Access: [batch, channel, rank, file]
                    if tensor[[0, c, h, w]] != 0.0 {
                         // c: channel, h: rank (0 is rank 1), w: file (0 is a)
                        println!("[{}, {}, {}]", c, h, w);
                    }
                }
            }
        }
    }
}