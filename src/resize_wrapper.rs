// Resize the input to fit into the network
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::aesthetic_predictor_conv::{AestheticPredictorConv, Config};

pub struct ResizeWrapper {
    model: AestheticPredictorConv,
    output_stride: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum Size {
    Size(usize, usize),
}

impl ResizeWrapper {
    pub fn new(vb: VarBuilder, c: &Config) -> Result<Self> {
        let model = AestheticPredictorConv::new(vb, c)?;

        Ok(ResizeWrapper {
            model,
            output_stride: 8,
        })
    }

    fn compute_sizes(&self, xs: &Tensor) -> (Size, Size) {
        let dims = xs.shape().clone().into_dims();
        let h_dim = dims.len() - 2;
        let w_dim = dims.len() - 1;
        let h = dims.get(h_dim).unwrap();
        let w = dims.get(w_dim).unwrap();
        let new_h = if h % self.output_stride != 0 {
            (h / self.output_stride) + 1
        } else {
            *h
        };
        let new_w = if w % self.output_stride != 0 {
            (w / self.output_stride) + 1
        } else {
            *w
        };

        (Size::Size(*h, *w), Size::Size(new_h, new_w))
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (old, new) = self.compute_sizes(xs);

        if old != new {
            let Size::Size(h, w) = new;
            self.model.forward(&xs.interpolate2d(h, w)?)
        } else {
            self.model.forward(&xs.to_owned())
        }
    }

    fn load_file<P: AsRef<std::path::Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        ResizeWrapper::new(vs, config)
    }
}
