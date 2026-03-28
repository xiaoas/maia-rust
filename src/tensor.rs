use ndarray::{Array3, ArrayViewMut2, Axis};
use shakmaty::{CastlingMode, Chess, Role, Setup, Square};

use crate::error::Error;

/// Data produced by the preprocessing step, ready for model consumption.
///
/// - `mirrored` tracks which positions were mirrored to
///   force the model's perspective to always be White.
/// - `chess_positions` holds the corresponding `Chess` objects after
///   any mirroring has been applied.  This is needed when converting
///   the model's output back into human-readable UCI moves.
pub struct PreprocessedData {
    /// Maia3 positions are represented from white-to-move perspective.
    pub mirrored: Vec<bool>,
    /// Potentially mirrored chess positions.
    pub chess_positions: Vec<Chess>,
}

/// Transform an iterator of `Setup`s into the input tensors
/// expected by the Maia3 model.
///
/// `batch_size` must match the number of setups provided; mismatches
/// will panic.  This function also records whether each position was
/// mirrored and returns the possibly‑mirrored `Chess` objects for
/// later use.
/// `tokens` has shape `[B, 64, 12]` where `B` is the batch size. Each
/// square stores one-hot piece channels in the order:
/// white P,N,B,R,Q,K then black p,n,b,r,q,k.
pub fn preprocess(
    setups: impl IntoIterator<Item = Setup>,
    batch_size: usize,
) -> Result<(Array3<f32>, PreprocessedData), Error> {
    let mut tokens = Array3::<f32>::zeros((batch_size, 64, 12));
    let mut mirrored_vec = Vec::with_capacity(batch_size);
    let mut chess_positions = Vec::with_capacity(batch_size);
    let mut last_index = 0;

    for (i, mut setup) in setups.into_iter().enumerate() {
        last_index = i;
        if i >= batch_size {
            panic!("More setups provided than batch size");
        }

        // If it's Black's turn we mirror so the network always sees
        // White-to-move positions.
        let mirrored = setup.turn.is_black();
        mirrored_vec.push(mirrored);

        if mirrored {
            setup.mirror();
        }

        board_to_tokens(&setup, tokens.index_axis_mut(Axis(0), i));
        let position: Chess = setup.position(CastlingMode::Standard)?;
        chess_positions.push(position);
    }

    if last_index + 1 != batch_size {
        panic!("Fewer setups provided than batch size");
    }

    Ok((
        tokens,
        PreprocessedData {
            chess_positions,
            mirrored: mirrored_vec,
        },
    ))
}

fn board_to_tokens(setup: &Setup, mut tokens: ArrayViewMut2<f32>) {
    for sq in Square::ALL {
        if let Some(piece) = setup.board.piece_at(sq) {
            let piece_idx = (if piece.color.is_white() { 0 } else { 6 })
                + match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };

            // Maia3 square index matches rank-major layout:
            // a1 => 0, b1 => 1, ..., h8 => 63.
            let square_idx = (sq.rank() as usize) * 8 + (sq.file() as usize);
            tokens[[square_idx, piece_idx]] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use shakmaty::fen::Fen;

    use super::*;

    #[test]
    fn verify_maia3_token_shape_and_content() {
        // Position: Start of Sicilian Defense (1. e4 c5)
        let fen_str = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
        let setup: Setup = fen_str.parse::<Fen>().unwrap().into_setup();

        // Process a single item
        let (tensor, _) = preprocess(vec![setup], 1).unwrap();
        assert_eq!(tensor.shape(), &[1, 64, 12]);

        // e4 white pawn should be one-hot in white pawn channel (0).
        let e4_idx = (3 * 8) + 4;
        assert_eq!(tensor[[0, e4_idx, 0]], 1.0);
    }
}
