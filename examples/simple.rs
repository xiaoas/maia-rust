use std::{fs, io::copy, path::Path};

use maia_rust::Maia;

const MODEL_URL: &str =
    "https://github.com/CSSLab/maia-platform-frontend/raw/refs/heads/main/public/maia3/maia3_simplified.onnx";
const MODEL_PATH: &str = "maia3_simplified.onnx";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Check and Download Model
    if !Path::new(MODEL_PATH).exists() {
        println!("Model not found at '{}'.", MODEL_PATH);
        println!("Downloading from {} ...", MODEL_URL);
        download_model(MODEL_URL, MODEL_PATH)?;
        println!("Download complete.");
    } else {
        println!("Found model at '{}'.", MODEL_PATH);
    }

    // 2. Initialize Maia
    // Note: The very first run might be slow as ORT initializes/optimizes the graph
    println!("Initializing Maia session...");
    let mut maia = Maia::from_file(MODEL_PATH)?;

    // 3. Define the position
    let start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // 4. Run Evaluation
    // We simulate a game between two 1500 Elo players
    println!("Evaluating starting position for 1500 vs 1500...");
    let result = maia.evaluate_fen(start_fen, 1500.0, 1500.0)?;

    // (If you prefer batched inference, the crate also exposes
    // `batch_evaluate`, an async variant `batch_evaluate_async`, and a
    // version that accepts `RunOptions`.)
    //
    // Example (synchronous):
    // let setups = vec![start_fen.parse::<shakmaty::fen::Fen>()?.into()];
    // let batch_results = maia.batch_evaluate(setups, &[1500.0], &[1500.0])?;

    // 5. Output Results
    println!("------------------------------------------------");
    println!("Win Probability (Side to move): {:.2}%", result.win * 100.0);
    println!("Draw Probability (Side to move): {:.2}%", result.draw * 100.0);
    println!("Loss Probability (Side to move): {:.2}%", result.loss * 100.0);
    println!("------------------------------------------------");
    println!("Top 5 Predicted Moves:");

    for (i, move_prob) in result.policy.iter().take(5).enumerate() {
        println!(
            "{}. {}  (Prob: {:.2}%)",
            i + 1,
            move_prob.uci,
            move_prob.probability * 100.0
        );
    }

    Ok(())
}

/// Helper to download file using reqwest (blocking)
fn download_model(url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = reqwest::blocking::get(url)?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: {}", response.status()).into());
    }

    let mut dest = fs::File::create(path)?;
    copy(&mut response, &mut dest)?;
    Ok(())
}
