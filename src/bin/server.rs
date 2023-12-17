use aesthetic_predictor_simple::AestheticPredictorSimple;
use candle_aesthetic_predictor::{parse_device, set_seed};
use candle_core::{DType, Device};
use clap::Parser;
use futures_util::TryStreamExt;
use std::sync::{Arc, RwLock};
use warp::Buf;
use warp::{filters::multipart::FormData, reject::Rejection, reply::Reply, Filter};
// use warp::Filter;
//
// use bytes::Buf;
// use futures::stream::TryStreamExt;
// use futures::Stream;
// use mime::Mime;
// use mpart_async::server::MultipartStream;
// use std::convert::Infallible;

#[path = "../aesthetic_predictor_simple.rs"]
mod aesthetic_predictor_simple;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model to use for predictions
    #[arg(short, long)]
    model: String,

    /// Seed (not usable on CPU)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Device (cpu, cuda, 0, 1, ...)
    #[arg(short, long)]
    device: Option<String>,
}

async fn predict(
    form: FormData,
    model: Arc<RwLock<AestheticPredictorSimple>>,
    device: Device,
) -> Result<impl Reply, Rejection> {
    println!("processing");
    let field_names: Vec<_> = form
        .and_then(|field| {
            let model = model.clone();
            let mut field = field;

            async move {
                let mut bytes: Vec<u8> = Vec::new();

                // field.data() only returns a piece of the content, you should call over it until it replies None
                while let Some(content) = field.data().await {
                    let content = content.unwrap();
                    let chunk: &[u8] = content.chunk();
                    bytes.extend_from_slice(chunk);
                }

                // let embedding_path = "/home/rockerboo/code/sd-ext/latents/00000-187894064808623f3f482a5ee531fe0bb9be5a54f5244d4a3-embedding.safetensors";
                let loaded = candle_core::safetensors::load_buffer(
                    &bytes,
                    &Device::cuda_if_available(0).unwrap(),
                )
                .unwrap();

                let logits = model
                    .read()
                    .unwrap()
                    .forward(loaded.get("embedding").unwrap())
                    .unwrap();

                println!("{:#?}", logits);
                println!("{:#?}", model.read().unwrap());

                Ok((
                    field.name().to_string(),
                    field.filename().unwrap().to_string(),
                    // format!(
                    //     "{:#?}",
                    logits
                        .squeeze(0)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap(),
                ))
            }
        })
        .try_collect()
        .await
        .unwrap();

    format!("{:?}", field_names);
    Ok::<_, warp::Rejection>(format!("{:#?}", field_names))
}

#[tokio::main]
async fn main() {
    let dtype = candle_core::DType::F32;

    let args = Args::parse();

    let device = parse_device(args.device).unwrap();

    set_seed(args.seed, &device).unwrap();

    // let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-10-223958/AestheticPredictorSE_ava_openai_clip_L14_5_128.safetensors";
    let weights_path = args.model;
    // let weights_path = args.model;
    println!("{}", weights_path);
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device).unwrap()
    };

    let model = aesthetic_predictor_simple::AestheticPredictorSimple::new(
        vs,
        &aesthetic_predictor_simple::Config {
            embed_dim: 768,
            in_channels: 1024,
        },
    )
    .unwrap();

    let model = Arc::new(RwLock::new(model));

    let model_filter = warp::any().map(move || model.clone());
    let device_filter = warp::any().map(move || device.clone());

    let route = warp::multipart::form().and_then(|form: FormData| async move {
        let field_names: Vec<_> = form
            .and_then(|mut field| async move {
                let mut bytes: Vec<u8> = Vec::new();

                // field.data() only returns a piece of the content, you should call over it until it replies None
                while let Some(content) = field.data().await {
                    let content = content.unwrap();
                    let chunk: &[u8] = content.chunk();
                    bytes.extend_from_slice(chunk);
                }

                let vs = candle_nn::VarBuilder::from_buffered_safetensors(
                    bytes,
                    DType::F32,
                    &Device::cuda_if_available(0).unwrap(),
                )
                .unwrap();

                let model = aesthetic_predictor_simple::AestheticPredictorSimple::new(
                    vs,
                    &aesthetic_predictor_simple::Config {
                        embed_dim: 768,
                        in_channels: 1024,
                    },
                )
                .unwrap();

                println!("{:?}", model);

                Ok("yo".to_string())
                // Ok((
                //     field.name().to_string(),
                //     field.filename().unwrap().to_string(),
                //     String::from_utf8_lossy(&*bytes).to_string(),
                // ))
            })
            .try_collect()
            .await
            .unwrap();

        Ok::<_, warp::Rejection>(format!("{:#?}", field_names))
    });

    // let handle_x = warp::multipart::form().and(device_filter.clone()).and(model.clone()).and_then(|form, device, model| async move {
    //
    //
    //
    // println!("{}", logits);
    // // dbg!(logits);
    // // format!("Hello, {}!", name);
    // Ok(warp::reply::json(
    //     &logits
    //         .squeeze(0)
    //         .unwrap()
    //         .squeeze(0)
    //         .unwrap()
    //         .to_scalar::<f32>()
    //         .unwrap()
    //     ))
    //
    //
    // });

    let readme = warp::get()
        .and(warp::path::end())
        .and(warp::fs::file("./index.html"));

    let model = warp::get()
        .and(warp::path("model.safetensors"))
        .and(warp::fs::file("./model.safetensors"));

    let hi = warp::path("hello")
        .and(warp::path::param())
        .and(warp::header("user-agent"))
        .map(|param: String, agent: String| format!("Hello {}, whose agent is {}", param, agent));

    let post_file = warp::post()
        .and(warp::path("predict"))
        .and(warp::multipart::form().max_length(50_000_000))
        .and(model_filter.clone())
        .and(device_filter.clone())
        .and_then(predict);

    // let handlers = hi.and(post_file);

    warp::serve(post_file.or(readme).or(hi).or(model))
        .run(([127, 0, 0, 1], 3030))
        .await;
}
