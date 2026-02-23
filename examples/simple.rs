use maia_rust::Maia;
use std::fs;
use std::io::copy;
use std::path::Path;

const MODEL_URL: &str = "https://github.com/CSSLab/maia-platform-frontend/raw/c2afee/public/maia2/maia_rapid.onnx";
const MODEL_PATH: &str = "maia_rapid.onnx";

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

    // 3. Define the position (Standard Start Position)
    let start_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
    
    // 4. Run Evaluation
    // We simulate a game between two 1500 Elo players
    println!("Evaluating starting position for 1500 vs 1500...");
    let result = maia.evaluate(start_fen, 1500, 1500)?;

    // 5. Output Results
    println!("------------------------------------------------");
    println!("Win Probability (Side to move): {:.2}%", result.value * 100.0);
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