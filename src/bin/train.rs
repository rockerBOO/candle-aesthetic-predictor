// use std::{fs::File, io::BufReader, path::Path};

use std::{fmt::Write, u32};

use aesthetic_predictor::model::aesthetic_predictor_simple;
use aesthetic_predictor::parse_device;
use aesthetic_predictor::set_seed;
use candle_core::error::Result;
use candle_core::Tensor;
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::arg;
use clap::Parser;
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CLIP input embedding
    #[arg(short, long)]
    embeddings: String,

    /// Ratings of the embeddings
    #[arg(short, long)]
    ratings: String,

    /// Learning rate
    #[arg(short, long)]
    lr: Option<f64>,

    // Weight decay
    #[arg(short, long)]
    weight_decay: Option<f64>,

    /// Batch size
    #[arg(short, long)]
    batch_size: Option<usize>,

    /// Epochs
    #[arg(long)]
    epochs: Option<usize>,

    /// Output file (model.safetensors)
    #[arg(short, long)]
    output: String,

    /// Seed (not usable on CPU)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Device (cpu, cuda, 0, 1, ...)
    #[arg(short, long)]
    device: Option<String>,
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let args = Args::parse();
    let dtype = candle_core::DType::F32;

    let device = parse_device(args.device)?;

    format!("Device: {:#?}", device);

    set_seed(args.seed, &device)?;

    format!("Seed: {:#?}", device);

    // let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-10-223958/AestheticPredictorSE_ava_openai_clip_L14_5_128.safetensors";
    // println!("{}", weights_path);
    // let vs =
    //     unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };

    // let embedding_path = "/home/rockerboo/code/sd-ext/latents/00000-187894064808623f3f482a5ee531fe0bb9be5a54f5244d4a3-embedding.safetensors";

    // let embedding = unsafe {
    //     candle_nn::VarBuilder::from_mmaped_safetensors(&[embedding_path], dtype, device)?
    // };

    // let loaded = candle_core::safetensors::load(embedding_path, &Device::cuda_if_available(0)?)?;

    // let embeddings_path =
    //     "/home/rockerboo/code/candle-aesthetic-predictor/ava_x_openai_clip_L14.npy";
    // let ratings_path = "/home/rockerboo/code/candle-aesthetic-predictor/ava_y_openai_clip_L14.npy";

    println!("Loading embeddings...");
    let embeddings = candle_core::Tensor::read_npy(args.embeddings)?
        .to_dtype(dtype)?
        .to_device(&device)?;

    println!("Loading ratings...");
    let ratings = candle_core::Tensor::read_npy(args.ratings)?
        .to_dtype(dtype)?
        .to_device(&device)?;
    // let embeddings = candle_core::npy(embeddings_path)?;
    // let ratings = candle_core::npy::NpzTensors::new(ratings_path)?;

    // dbg!(embeddings.names());
    // dbg!(ratings);

    println!("Loading weights...");
    let varmap = VarMap::new();
    // let vs_blank = VarBuilder::zeros(dtype, device);
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    // let model = linear(2, 1, vb.pp("linear"))?;
    //
    println!("Loading model...");
    let model = aesthetic_predictor_simple::AestheticPredictorSimple::new(
        vb,
        &aesthetic_predictor_simple::Config {
            embed_dim: 768,
            in_channels: 1024,
        },
    )?;

    let lr = args.lr.unwrap_or(1e-2);
    let weight_decay = args.weight_decay.unwrap_or(0.1);
    let beta1 = 0.9;
    let beta2 = 0.999;
    let params = ParamsAdamW {
        lr,
        weight_decay,
        beta1,
        beta2,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    println!("Using AdamW optimizer with {} lr {} weight decay {} beta1 {} beta2  steps", lr, weight_decay, beta1, beta2);

    let epochs = args.epochs.unwrap_or(100);

    let mut losses: Vec<f32> = vec![];

    let batch_size = args.batch_size.unwrap_or(32768);

    let total_results = embeddings.dims()[0];
    // let batch_arr: [f32; 6] = [0.; 6];
    let indices = Vec::from_iter(0..total_results);
    let batches = total_results / batch_size;
    let total_steps = batches * epochs;

    // let mut scheduler = candle_scheduler::OneCycle::new(1e-3, 0.9, 25., total_steps);
    let mut scheduler =
        candle_scheduler::CosineAnnealing::new(args.lr.unwrap_or(1e-3), total_steps, 1e-6);

    println!("Using CosineAnnealing scheduler with {} lr, at {} steps", args.lr.unwrap_or(1e-3), total_steps);

    let pb = ProgressBar::new((total_steps).try_into().unwrap());
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {wide_msg} {human_pos}/{human_len} {per_sec} ({eta})")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}h", state.eta().as_secs_f64() / 60. / 60.).unwrap())
        .progress_chars("#>-"));

    println!("Starting training...");
    println!("{} epochs {} batches {} per batch", epochs, batches, batch_size);
    for epoch in 0..epochs {
        for i in 0..batches {
            // if i % epochs == 0 {
            //     println!("{}", if i > 0 { (epochs / i) + 1 } else { 1 });
            // }
            // for (step, (embedding, rating)) in embeddings.get(0)?.zip(ratings.get(0)?).enumerate() {
            // let embedding = embeddings.get(name)?;
            // let rating = ratings.get(rating_name)?;

            // let embedding = loaded.get("embedding").unwrap();
            let batch_indices =
                Tensor::from_iter(indices.iter().take(batch_size).map(|v| *v as u32), &device)?;
            let batched_embeddings = embeddings.index_select(&batch_indices, 0)?.unsqueeze(0)?;
            let batched_ratings = ratings.index_select(&batch_indices, 0)?.unsqueeze(0)?;

            let logits = model.forward(&batched_embeddings)?;
            // println!("{} {}", &logits, ratings.get(i)?);
            // let loss = loss::mse(&logits, &ratings.get(i)?.unsqueeze(0)?)?;
            let loss = loss::mse(&logits, &batched_ratings)?;

            opt.backward_step(&loss)?;
            scheduler.step(&mut opt);
            losses.push(loss.to_vec0::<f32>()?);
            pb.set_message(format!(
                "{} epoch {} step {} lr {} loss",
                HumanCount((epoch + 1) as u64),
                HumanCount(i as u64),
                scheduler.get_lr(),
                losses.iter().sum::<f32>() / losses.len() as f32
            ));
            pb.inc(1);
        }
    }
    pb.finish_and_clear();

    let model_file = args.output;
    println!("Saving to {}", model_file);

    varmap.save(model_file)?;

    Ok(())
}
