use ndarray::{ArrayView1, Axis};
use ort::{session::Session, value::Tensor};
use shakmaty::{Chess, Position, Setup};

use crate::{
    error::MaiaError,
    moves::ALL_MOVES,
    tensor::{map_elos_to_categories, preprocess},
    types::{EvaluationResult, MoveProbability},
};

pub struct Maia {
    session: Session,
}

impl Maia {
    /// Initialize from a local `.onnx` file path
    pub fn from_file(path: &str) -> Result<Self, MaiaError> {
        let session = Session::builder()?.commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Initialize from raw bytes
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, MaiaError> {
        let session = Session::builder()?.commit_from_memory(model_bytes)?;

        Ok(Self { session })
    }

    pub fn evaluate(
        &mut self,
        fen: &str,
        elo_self: u32,
        elo_oppo: u32,
    ) -> Result<EvaluationResult, MaiaError> {
        let fen: shakmaty::fen::Fen = fen.parse()?;
        let setup: Setup = fen.into();

        let results = self.batch_evaluate([setup], &[elo_self], &[elo_oppo])?;
        Ok(results.into_iter().next().unwrap())
    }
    /// the slices should all have the same length, representing the batch size.
    /// Maia2 supports elo buckets from 10xx to 20xx in 100-point increments. Values will be clamped to the nearest bucket (e.g., 900 -> 1000 bucket, 1170 -> 1100 bucket, 2050 -> 2000 bucket).
    pub fn batch_evaluate(
        &mut self,
        setups: impl IntoIterator<Item = Setup>,
        elo_selfs: &[u32],
        elo_oppos: &[u32],
    ) -> Result<Vec<EvaluationResult>, MaiaError> {
        let batch_size = elo_selfs.len();
        assert_eq!(elo_oppos.len(), batch_size);

        // 1. Preprocess
        let data = preprocess(setups, batch_size)?;

        // 2. Convert source to inputs
        let elo_selfs = map_elos_to_categories(elo_selfs);
        let elo_oppos = map_elos_to_categories(elo_oppos);
        // 3. Run Inference
        let outputs = self.session.run(ort::inputs! {
                "boards" => Tensor::from_array(data.board_tensor)?,
                "elo_self" => Tensor::from_array(([batch_size], elo_selfs))?,
                "elo_oppo" => Tensor::from_array(([batch_size], elo_oppos))?,
        })?;

        // 4. Extract Logits
        let logits_maia = outputs["logits_maia"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap(); // logits_maia should be 2D

        let logits_value = outputs["logits_value"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix1>()
            .unwrap();

        // 5. Postprocess into EvaluationResults
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let logits_for_item = logits_maia.index_axis(Axis(0), i);
            let raw_value = logits_value[[i]];

            let result = Self::process_output(
                logits_for_item,
                raw_value,
                &data.chess_positions[i],
                data.mirrored[i],
            );
            results.push(result);
        }

        Ok(results)
    }

    fn process_output(
        logits_maia: ArrayView1<f32>,
        raw_value: f32,
        chess: &Chess,
        mirrored: bool,
    ) -> EvaluationResult {
        // Compute side-to-move's winning probability
        let mut win_prob = (raw_value / 2.0 + 0.5).clamp(0.0, 1.0);
        if mirrored {
            win_prob = 1.0 - win_prob;
        }

        let legal_moves = chess.legal_moves();

        let mut max_logit = f32::NEG_INFINITY;
        let mut move_data = Vec::with_capacity(legal_moves.len());

        // for (uci, &idx) in &*ALL_MOVES {
        //     let actual_uci = if mirrored { uci.to_mirrored() } else { *uci };
        //     let logit = logits_maia[idx];

        //     if logit > max_logit {
        //         max_logit = logit;
        //     }
        //     move_data.push((actual_uci, logit));
        // }
        for m in &legal_moves {
            // Get the actual legal move in UciMove format
            let uci = m.to_uci(shakmaty::CastlingMode::Standard);

            // If the board was mirrored (Black's turn), use the mirrored UCI in response
            let actual_uci = if mirrored {
                uci.to_mirrored()
            } else {
                uci
            };

            // Retrieve logit for the move according to the model's perspective
            if let Some(&idx) = ALL_MOVES.get(&uci) {
                let logit = logits_maia[idx];

                if logit > max_logit {
                    max_logit = logit;
                }

                // Track the actual unmirrored UciMove paired with its logit
                move_data.push((actual_uci, logit));
            }
        }

        // Apply Softmax
        let mut sum_exp = 0.0;
        let mut exps = Vec::with_capacity(move_data.len());

        for &(_, logit) in &move_data {
            let exp = (logit - max_logit).exp();
            sum_exp += exp;
            exps.push(exp);
        }

        // Create MoveProbability
        let mut policy = Vec::with_capacity(move_data.len());
        for (i, (uci, _)) in move_data.into_iter().enumerate() {
            policy.push(MoveProbability {
                uci,
                probability: exps[i] / sum_exp,
            });
        }

        // Sort by descending probability
        policy.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        EvaluationResult {
            policy,
            value: win_prob,
        }
    }
}
