use std::path::Path;

use ndarray::{ArrayView1, Axis};
use ort::{session::Session, value::Tensor};
use shakmaty::{Chess, Position, Setup};

use crate::{
    error::Error,
    moves::ALL_MOVES,
    tensor::preprocess,
    types::{EvaluationResult, MoveProbability},
};

/// Wrapper around an ONNX Runtime session configured with the
/// Maia3 model.
///
/// The struct manages an inference session and exposes evaluation
/// functions that accept FEN strings or `shakmaty` setups.  The
/// model expects inputs in a specific tensor layout; helper functions
/// in the `tensor` module handle the conversion.
pub struct Maia {
    session: Session,
}

impl Maia {
    /// Create a Maia instance by loading a model from a `.onnx` file.
    ///
    /// # Errors
    /// Returns an [`Error::OrtError`] if the session cannot be
    /// constructed or the file cannot be read.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let session = Session::builder()?.commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Construct from raw ONNX model bytes, useful for embedding the
    /// model in the binary or loading from a network source.
    ///
    /// # Errors
    /// Similar to [`from_file`], errors are propagated as
    /// [`Error::OrtError`].
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, Error> {
        let session = Session::builder()?.commit_from_memory(model_bytes)?;

        Ok(Self { session })
    }

    /// Construct from an existing ONNX Runtime session that's running Maia3, allowing users to
    /// configure the session themselves.
    pub fn from_session(session: Session) -> Self {
        Self { session }
    }
    /// Evaluate a single position specified by FEN.
    ///
    /// ELO values for both sides are provided as raw floating-point
    /// inputs used by Maia3's continuous conditioning.
    ///
    /// # Errors
    /// - Returns [`Error::InvalidFen`] if the FEN cannot be parsed.
    /// - Propagates any errors from batched evaluation.
    pub fn evaluate_fen(
        &mut self,
        fen: &str,
        elo_self: f32,
        elo_oppo: f32,
    ) -> Result<EvaluationResult, Error> {
        let fen: shakmaty::fen::Fen = fen.parse()?;
        let setup: Setup = fen.into();

        let results = self.batch_evaluate([setup], &[elo_self], &[elo_oppo])?;
        Ok(results.into_iter().next().unwrap())
    }
    /// Evaluate a batch of positions simultaneously.
    ///
    /// The iterator of [`Setup`]s supplies the board states; the slices of
    /// `elo_selfs` and `elo_oppos` must have identical length equal to the
    /// number of setups. Batch evaluation is significantly faster than
    /// calling [`evaluate_fen`] repeatedly when performing multiple inferences.
    ///
    /// # Errors
    /// See [`Error`] for possible failure modes.
    pub fn batch_evaluate(
        &mut self,
        setups: impl IntoIterator<Item = Setup>,
        elo_selfs: &[f32],
        elo_oppos: &[f32],
    ) -> Result<Vec<EvaluationResult>, Error> {
        let batch_size = elo_selfs.len();
        assert_eq!(elo_oppos.len(), batch_size);

        // 1. Preprocess
        let (board, data) = preprocess(setups, batch_size)?;

        // 3. Run inference and postprocess
        let outputs = self.session.run(ort::inputs! {
                "tokens" => Tensor::from_array(board)?,
                "elo_self" => Tensor::from_array(([batch_size], elo_selfs.to_vec()))?,
                "elo_oppo" => Tensor::from_array(([batch_size], elo_oppos.to_vec()))?,
        })?;

        Self::finalize_batch(outputs, batch_size, data)
    }

    /// Asynchronous version of [`batch_evaluate`].
    ///
    /// This function behaves identically to `batch_evaluate`, except that
    /// it uses [`Session::run_async`] internally and therefore returns a
    /// future that must be `.await`ed.  It is useful when the caller is
    /// already running inside an async runtime and wants to avoid blocking.
    pub async fn batch_evaluate_async(
        &mut self,
        setups: impl IntoIterator<Item = Setup>,
        elo_selfs: &[f32],
        elo_oppos: &[f32],
        options: &ort::session::RunOptions,
    ) -> Result<Vec<EvaluationResult>, Error> {
        let batch_size = elo_selfs.len();
        assert_eq!(elo_oppos.len(), batch_size);

        // 1. Preprocess
        let (board, data) = preprocess(setups, batch_size)?;

        // 3. Run inference asynchronously and postprocess
        let outputs = self
            .session
            .run_async(
                ort::inputs! {
                    "tokens" => Tensor::from_array(board)?,
                    "elo_self" => Tensor::from_array(([batch_size], elo_selfs.to_vec()))?,
                    "elo_oppo" => Tensor::from_array(([batch_size], elo_oppos.to_vec()))?,
                },
                &options,
            )?
            .await?;

        Self::finalize_batch(outputs, batch_size, data)
    }

    /// Batch evaluation that allows callers to supply custom `RunOptions`.
    ///
    /// The provided [`ort::session::RunOptions`] are forwarded directly to
    /// [`Session::run_with_options`].  This is handy when the user wants to
    /// adjust logging, threading, or profiling behaviour on a per-inference
    /// basis.
    pub fn batch_evaluate_with_options(
        &mut self,
        setups: impl IntoIterator<Item = Setup>,
        elo_selfs: &[f32],
        elo_oppos: &[f32],
        options: &ort::session::RunOptions,
    ) -> Result<Vec<EvaluationResult>, Error> {
        let batch_size = elo_selfs.len();
        assert_eq!(elo_oppos.len(), batch_size);

        // 1. Preprocess
        let (board, data) = preprocess(setups, batch_size)?;

        // 3. Run inference with options and postprocess
        let outputs = self.session.run_with_options(
            ort::inputs! {
                "tokens" => Tensor::from_array(board)?,
                "elo_self" => Tensor::from_array(([batch_size], elo_selfs.to_vec()))?,
                "elo_oppo" => Tensor::from_array(([batch_size], elo_oppos.to_vec()))?,
            },
            options,
        )?;

        Self::finalize_batch(outputs, batch_size, data)
    }

    /// Internal helper used by the various batch evaluation entrypoints.
    ///
    /// Takes ownership of the preprocessed data so that we can consume it
    /// without having to duplicate the logic in `batch_evaluate`,
    /// `batch_evaluate_async` and `batch_evaluate_with_options`.
    fn finalize_batch(
        outputs: ort::session::SessionOutputs,
        batch_size: usize,
        data: crate::tensor::PreprocessedData,
    ) -> Result<Vec<EvaluationResult>, Error> {
        // 4. Extract Logits
        let logits_move = outputs["logits_move"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap(); // logits_move should be 2D

        let logits_value = outputs["logits_value"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap(); // logits_value should be [batch, 3]

        // 5. Postprocess into EvaluationResults
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let logits_for_item = logits_move.index_axis(Axis(0), i);
            let raw_wdl = logits_value.index_axis(Axis(0), i);

            let result = Self::process_output(
                logits_for_item,
                raw_wdl,
                &data.chess_positions[i],
                data.mirrored[i],
            );
            results.push(result);
        }

        Ok(results)
    }

    /// Convert raw model outputs to a structured [`EvaluationResult`].
    ///
    /// `logits_move` contains unnormalized policy logits for all moves
    /// in the fixed Maia3 vocabulary. `raw_wdl` is a 3-logit vector
    /// ordered as loss/draw/win from side-to-move perspective.
    fn process_output(
        logits_move: ArrayView1<f32>,
        raw_wdl: ArrayView1<f32>,
        chess: &Chess,
        mirrored: bool,
    ) -> EvaluationResult {
        // Convert L/D/W logits to side-to-move probabilities.
        let max_wdl = raw_wdl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_l = (raw_wdl[0] - max_wdl).exp();
        let exp_d = (raw_wdl[1] - max_wdl).exp();
        let exp_w = (raw_wdl[2] - max_wdl).exp();
        let sum_wdl = exp_l + exp_d + exp_w;
        let mut loss_prob = exp_l / sum_wdl;
        let draw_prob = exp_d / sum_wdl;
        let mut win_prob = exp_w / sum_wdl;
        if mirrored {
            // Mirroring swaps side-to-move perspective, so win/loss invert.
            std::mem::swap(&mut win_prob, &mut loss_prob);
        }

        let legal_moves = chess.legal_moves();

        let mut max_logit = f32::NEG_INFINITY;
        let mut move_data = Vec::with_capacity(legal_moves.len());

        // for (uci, &idx) in &*ALL_MOVES {
        //     let actual_uci = if mirrored { uci.to_mirrored() } else { *uci };
        //     let logit = logits_move[idx];

        //     if logit > max_logit {
        //         max_logit = logit;
        //     }
        //     move_data.push((actual_uci, logit));
        // }
        for m in &legal_moves {
            // Convert the `shakmaty` move into the UCI notation the model
            // expects.
            let uci = m.to_uci(shakmaty::CastlingMode::Standard);

            // Look up the move's index in the fixed vocabulary.
            if let Some(&idx) = ALL_MOVES.get(&uci) {
                let logit = logits_move[idx];

                if logit > max_logit {
                    max_logit = logit;
                }

                // If input was mirrored (because it was Black's turn), we
                // must mirror the move back when reporting results.
                let actual_uci = if mirrored { uci.to_mirrored() } else { uci };
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
            win: win_prob,
            draw: draw_prob,
            loss: loss_prob,
        }
    }
}

#[cfg(test)]
mod tests {
    use ort::logging::LogLevel;
    use shakmaty::fen::Fen;

    use super::*;

    fn sample_setup() -> Setup {
        let fen: Fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
            .parse()
            .unwrap();
        fen.into()
    }

    #[test]
    #[ignore = "requires local Maia3 ONNX model file"]
    fn sync_and_options_evaluate() {
        let mut maia = Maia::from_file("maia3_simplified.onnx").expect("load model");
        let setups = vec![sample_setup()];

        let r1 = maia
            .batch_evaluate(setups.clone(), &[1500.0], &[1500.0])
            .expect("sync eval");
        assert_eq!(r1.len(), 1);

        let mut opts = ort::session::RunOptions::new().unwrap();
        opts.set_log_level(LogLevel::Warning).unwrap();
        let r2 = maia
            .batch_evaluate_with_options(setups, &[1500.0], &[1500.0], &opts)
            .expect("options eval");
        assert_eq!(r2.len(), 1);
    }

    #[tokio::test]
    #[ignore = "requires local Maia3 ONNX model file"]
    async fn async_evaluate() {
        let mut maia = Maia::from_file("maia3_simplified.onnx").expect("load model");
        let setups = vec![sample_setup()];

        let opts = ort::session::RunOptions::new().unwrap();
        let r = maia
            .batch_evaluate_async(setups, &[1500.0], &[1500.0], &opts)
            .await
            .expect("async eval");
        assert_eq!(r.len(), 1);
    }
}
