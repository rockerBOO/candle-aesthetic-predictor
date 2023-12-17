use candle_core::{Device, Result};
use clap::Parser;

#[path = "../aesthetic_predictor_simple.rs"]
mod aesthetic_predictor_simple;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CLIP input embedding
    #[arg(short, long)]
    input: String,

    /// Model to use for predictions
    #[arg(short, long)]
    model: String,
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let args = Args::parse();

    let dtype = candle_core::DType::F32;
    let device = &Device::cuda_if_available(0)?;

    // let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-10-223958/AestheticPredictorSE_ava_openai_clip_L14_5_128.safetensors";
    let weights_path = args.model;
    println!("{}", weights_path);
    let vs =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };

    let model = aesthetic_predictor_simple::AestheticPredictorSimple::new(
        vs,
        &aesthetic_predictor_simple::Config {
            embed_dim: 768,
            in_channels: 1024,
        },
    )?;

    // let embedding_path = "/home/rockerboo/code/sd-ext/latents/00000-187894064808623f3f482a5ee531fe0bb9be5a54f5244d4a3-embedding.safetensors";
    let embedding_path = args.input;

    // let embedding = unsafe {
    //     candle_nn::VarBuilder::from_mmaped_safetensors(&[embedding_path], dtype, device)?
    // };

    let loaded = candle_core::safetensors::load(embedding_path, &Device::cuda_if_available(0)?)?;

    let logits = model.forward(loaded.get("embedding").unwrap())?;

    println!("{}", logits);
    dbg!(logits);

    Ok(())
}
