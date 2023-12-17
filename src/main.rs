mod aesthetic_predictor_conv;
mod aesthetic_predictor_simple;
mod resize_wrapper;

use candle_core::{DType, Device, Module, Result, Tensor, D};

use candle_nn::VarBuilder;
// use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae;
// use crate::aesthetic_predictor_conv::AestheticPredictorConv;

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

fn main() -> Result<()> {
    // let tensor = Tensor::randn(0.0_f32, 1.0_f32, (1, 768), &Device::cuda_if_available(0)?)?;
    let dtype = candle_core::DType::F32;
    let device = &Device::cuda_if_available(0)?;
    // let device = &Device::Cpu;

    let latents = {
        let vae_cfg = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };

        let vae_path = "/mnt/900/vae/stable-diffusion-v1-5-vae.safetensors";
        let vae_vs =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, device)? };

        let vae = vae::AutoEncoderKL::new(vae_vs, 3, 3, vae_cfg)?;

        // let image_path = "/home/rockerboo/art/boo/00001-1297897296.png";
        let image_path = "/home/rockerboo/art/boo/00012-1532577046.png";
        let image_tensor = image_preprocess(image_path)
            .unwrap()
            .to_device(device)?
            .to_dtype(dtype)?;

        vae.encode(&image_tensor)
    };

    // let latents_path = "/home/rockerboo/code/sd-ext/latents/00000-3391699349-latent.safetensors";

    // let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-10-223958/AestheticPredictorSE_ava_openai_clip_L14_5_128.safetensors";
    // let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-11-191601/ResizeWrapper_ava_openai_clip_L14_5_1.safetensors";
    let weights_path = "/home/rockerboo/code/sd-ext/training/sets/2023-12-11-194450/ResizeWrapper_ava_openai_clip_L14_5_1.safetensors";

    // let embedding_path = "/home/rockerboo/code/sd-ext/latents/00000-187894064808623f3f482a5ee531fe0bb9be5a54f5244d4a3-embedding.safetensors";
    println!("{}", weights_path);
    let vs =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };

    // let vs_blank = VarBuilder::zeros(dtype, device);

    // let vb = VarBuilder::from_buffered_safetensors(weights_buffer, DType::F32, &Device::Cpu)?;

    // let model = aesthetic_predictor_simple::AestheticPredictorSimple::new(
    //     vs,
    //     &aesthetic_predictor_simple::Config {},
    // )?;
    //vs.pp("model")
    let wrapper = resize_wrapper::ResizeWrapper::new(vs, &aesthetic_predictor_conv::Config {})?;

    // let loaded =
    //     candle_core::safetensors::load(latents_path, &Device::cuda_if_available(0)?)?;

    // let embedding = loaded.get("latent");
    let logits = wrapper.forward(&latents?.sample()?)?;

    // logits.squeeze(0)?.to_vec1::<f32>()?.iter().for_each(|v| {
    //     dbg!(v);
    // });
    // let _ = logits.to_scalar::<f32>();

    dbg!(logits);

    Ok(())
}
