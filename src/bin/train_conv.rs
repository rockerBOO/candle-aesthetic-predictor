use std::clone::Clone;
use std::fmt::Display;
use std::fmt::Write;

use aesthetic_predictor::latent::vae;
use aesthetic_predictor::model::aesthetic_predictor_conv;
use aesthetic_predictor::parse_device;
use aesthetic_predictor::set_seed;
use aesthetic_predictor::AestheticPredictorError;
use aesthetic_predictor::Result;
use candle_core::Tensor;
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory with training images
    #[arg(short, long)]
    images_dir: String,

    // /// Ratings of the embeddings
    // #[arg(short, long)]
    // ratings: String,
    /// Learning rate
    #[arg(long)]
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

fn collate_images<T: Display + AsRef<std::path::Path> + std::fmt::Debug>(
    vae: &vae::Vae,
    dir: &str,
    images: Vec<T>,
) -> Result<Vec<Result<Tensor>>> {
    let results: Vec<Result<Tensor>> = images
        .iter()
        .map(|image_path| {
            dbg!(image_path);
            Ok(
                aesthetic_predictor::latent::image::image_preprocess(format!(
                    "{}/{}",
                    dir, image_path
                ))?
                .to_device(vae.device())?
                .to_dtype(*vae.dtype())?,
            )
        })
        .collect();

    if results.iter().any(|v| v.is_err()) {
        println!("{:#?}", results);
        // results.iter().map(|r| {
        //     r.map_err(|e| println!("{:#?}", e)).unwrap_err()
        // });
        Err(AestheticPredictorError::Msg(
            "Could not load all the images".to_string(),
        ))
    } else {
        Ok(results)
    }
}

const EXTS: [&str; 6] = ["png", "jpg", "jpeg", "webp", "bmp", "avif"];
fn filter_images(d: &std::fs::DirEntry) -> bool {
    EXTS.iter()
        .any(|ext| d.file_name().into_string().unwrap().contains(ext))
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let args = Args::parse();
    let dtype = candle_core::DType::F32;
    format!("Dtype: {:#?}", dtype);

    let device = parse_device(args.device)?;
    format!("Device: {:#?}", device);

    set_seed(args.seed, &device)?;
    format!("Seed: {:#?}", device);

    println!("Loading latents...");
    // let latents = candle_core::Tensor::read_npy(args.latents)?
    //     .to_dtype(dtype)?
    //     .to_device(&device)?;

    let images_dir = "/home/rockerboo/art/boo";
    let images = std::fs::read_dir(images_dir)?;

    // println!("Loading ratings...");
    // let ratings = candle_core::Tensor::read_npy(args.ratings)?
    //     .to_dtype(dtype)?
    //     .to_device(&device)?;

    let vae_path = "/mnt/900/vae/stable-diffusion-v1-5-vae.safetensors";
    let vae = vae::Vae::new_from_file(vae_path, dtype, &device)?;

    println!("Loading weights...");
    let varmap = VarMap::new();
    // let vs_blank = VarBuilder::zeros(dtype, device);
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    println!("Loading model...");
    let model = aesthetic_predictor_conv::AestheticPredictorConv::new(
        vb,
        &aesthetic_predictor_conv::Config {
            in_channels: 4,
            out_channels: 32,
        },
    )?;

    println!("{:#?}", model);

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

    println!(
        "Using AdamW optimizer with {} lr {} weight decay {} beta1 {} beta2  steps",
        lr, weight_decay, beta1, beta2
    );

    let epochs = args.epochs.unwrap_or(100);

    let mut losses: Vec<f32> = vec![];

    let batch_size = args.batch_size.unwrap_or(1);

    // let total_results = latents.dims()[0];
    let total_results = images.filter_map(|v| v.ok()).filter(filter_images).count();

    // let indices = Vec::from_iter(0..total_results);
    let batches = total_results / batch_size;
    let total_steps = batches * epochs;

    // let mut scheduler = candle_scheduler::OneCycle::new(1e-3, 0.9, 25., total_steps);
    let mut scheduler =
        candle_scheduler::CosineAnnealing::new(args.lr.unwrap_or(1e-3), total_steps, 1e-6);

    println!(
        "Using CosineAnnealing scheduler with {} lr, at {} steps",
        args.lr.unwrap_or(1e-3),
        total_steps
    );

    let pb = ProgressBar::new((total_steps).try_into().unwrap());
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {wide_msg} {human_pos}/{human_len} {per_sec} ({eta})")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}h", state.eta().as_secs_f64() / 60. / 60.).unwrap())
        .progress_chars("#>-"));

    println!("Starting training...");
    println!(
        "{} epochs {} batches {} per batch",
        epochs, batches, batch_size
    );
    for epoch in 0..epochs {
        for i in 0..batches {
            // let batch_indices =
            //     Tensor::from_iter(indices.iter().take(batch_size).map(|v| *v as u32), &device)?;

            // let images_batch: Vec<std::result::Result<std::fs::DirEntry, io::Error>> =
            //     .filter(|e| e.is_ok()).collect();
            let batch_images = collate_images(
                &vae,
                images_dir,
                std::fs::read_dir(images_dir)?
                    .filter_map(|v| v.ok())
                    .filter(filter_images)
                    .skip(i * batch_size)
                    .take(batch_size)
                    .filter_map(|entry| {
                        entry
                            .path()
                            .file_name()
                            .and_then(|n| n.to_str().map(String::from))
                    })
                    .collect(),
            );

            if let Err(err) = batch_images {
                println!("{:#?}", err);
                continue;
            }

            let b_images: Vec<Tensor> = batch_images?.into_iter().filter_map(|i| i.ok()).collect();

            let batched_latents = vae.encode(&b_images)?;

            // let batched_latents = latents.index_select(&batch_indices, 0)?.unsqueeze(0)?;
            // let batched_ratings = ratings.index_select(&batch_indices, 0)?.unsqueeze(0)?;
            let random_ratings = Tensor::randn(0_f32, 10., (batch_size, 1), &device)?;

            let logits = model.forward(&batched_latents.sample()?)?;
            println!("{} {}", &logits, random_ratings);
            let loss = loss::mse(&logits, &random_ratings.unsqueeze(0)?.unsqueeze(0)?)?;

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
