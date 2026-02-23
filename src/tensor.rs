use ndarray::{Array4, ArrayViewMut3, Axis};
use shakmaty::{CastlingMode, Chess, Role, Setup, Square};

use crate::error::MaiaError;

pub struct PreprocessedData {
    pub board_tensor: Array4<f32>, // Shape: [B, 18, 8, 8]
    pub chess_positions: Vec<Chess>,
    /// maia2 trains on positions from the perspective of white. Black positions are mirrored before being fed into the model.
    pub mirrored: Vec<bool>,
}

/// Computes the Elo bucket.
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
    // 1. Piece Placement (Channels 0..11)
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

    // 2. Player's turn (Channel 12)
    tensor
        .index_axis_mut(Axis(0), 12)
        .fill(setup.turn.is_white() as u8 as f32); // 1.0 if is_white else 0.0

    // 3. Castling rights (Channels 13..16)
    // Bitboard efficiently tracks the original rook squares for standard castling mappings
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

    // 4. En passant target (Channel 17)
    if let Some(ep_sq) = setup.ep_square {
        tensor[[17, ep_sq.rank() as usize, ep_sq.file() as usize]] = 1.0;
    }
}
