use crate::Result;

pub fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> Result<candle_core::Tensor> {
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
    let img = candle_core::Tensor::from_vec(img, (height, width, 3), &candle_core::Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(candle_core::DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

// let image_tensors: Vec<Result<&candle_core::Tensor>> = images
//     .iter()
//     .map(|image_path| {
//         image_preprocess(image_path)?
//             .to_device(self.device)?
//             .to_dtype(self.dtype)?
//     })
//     .collect();
//         AestheticPredictorError::VaeInvalidImages {
//             errors: vec![],
//         },
//         |e, err| match e {
//             AestheticPredictorError::Candle(e) => ,
//             AestheticPredictorError::VaeInvalidImages { errors } => todo!(),
//         },
//     );
