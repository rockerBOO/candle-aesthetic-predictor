mod predictor;

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let xs = Tensor::arange::<f32>(0., 64., &Device::Cpu)?.reshape((1, 64, 1, 1))?;

    let xs = xs.max_pool2d(2)?;
    println!("{}", xs);

    Ok(())
}
