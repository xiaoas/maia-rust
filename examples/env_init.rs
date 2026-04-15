//! Print which allocator/device ONNX Runtime selected for a session.
//!
//! Run with: `cargo run --example session_device`
 
use std::{fs, io::copy, path::Path};
 
use ort::{environment::Environment, session::Session};
 
const MODEL_URL: &str =
    "https://github.com/CSSLab/maia-platform-frontend/raw/refs/heads/main/public/maia3/maia3_simplified.onnx";
const MODEL_PATH: &str = "maia3_simplified.onnx";
 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(MODEL_PATH).exists() {
        println!("Model not found at '{}'. Downloading...", MODEL_PATH);
        download_model(MODEL_URL, MODEL_PATH)?;
        println!("Download complete.");
    } else {
        println!("Found model at '{}'.", MODEL_PATH);
    }
 
    // explcit init
    ort::init().commit();

    let env = Environment::current()?;
    for device in env.devices() {
        println!(
            "{id} ({vendor} {ty:?} - {ep})",
            id = device.id(),
            vendor = device.vendor()?,
            ty = device.ty(),
            ep = device.ep()?
        );
    }

    // Build a regular session using the default builder settings.
    // let session = Session::builder()?
    // .with_execution_providers([ep::CoreML::default().build()])?
    // .commit_from_file(MODEL_PATH)?;
    let session = Session::builder()?.commit_from_file(MODEL_PATH)?;
 
    let allocator = session.allocator();
    let memory_info = allocator.memory_info();
 
    println!("Session allocator memory info:");
    println!("  allocation_device: {:?}", memory_info.allocation_device());
    println!("  device_id: {}", memory_info.device_id());
    println!("  allocator_type: {:?}", memory_info.allocator_type());
    println!("  memory_type: {:?}", memory_info.memory_type());
    println!("  device_type: {:?}", memory_info.device_type());
    println!("  is_cpu_accessible: {}", memory_info.is_cpu_accessible());
 
    Ok(())
}
 
fn download_model(url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = reqwest::blocking::get(url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download: {}", response.status()).into());
    }
    let mut dest = fs::File::create(path)?;
    copy(&mut response, &mut dest)?;
    Ok(())
}