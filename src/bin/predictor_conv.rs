use aesthetic_predictor::model::aesthetic_predictor_conv;
use candle_core::{Device, Result};
use clap::Parser;

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

    let weights_path = args.model;
    println!("{}", weights_path);
    let vs =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };

    let model = aesthetic_predictor_conv::AestheticPredictorConv::new(
        vs,
        &aesthetic_predictor_conv::Config {
            in_channels: 4,
            out_channels: 1,
        },
    )?;

    let embedding_path = args.input;
    let loaded = candle_core::safetensors::load(embedding_path, &Device::cuda_if_available(0)?)?;

    let logits = model.forward(loaded.get("latents").unwrap())?;

    println!("{}", logits);
    dbg!(logits);

    Ok(())
}
