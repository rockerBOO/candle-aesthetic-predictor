use std::io;

#[derive(thiserror::Error, Debug)]
pub enum AestheticPredictorError {
    #[error("candle_core error")]
    Candle(candle_core::Error),

    #[error("Image error")]
    Image(image::ImageError),

    #[error("IO Error")]
    Io(io::Error),

    #[error("Error: {0:?}")]
    Msg(String)
}

impl From<candle_core::Error> for AestheticPredictorError {
    fn from(value: candle_core::Error) -> Self {
        AestheticPredictorError::Candle(value)
    }
}

impl From<image::ImageError> for AestheticPredictorError {
    fn from(value: image::ImageError) -> Self {
        AestheticPredictorError::Image(value)
    }
}

impl From<io::Error> for AestheticPredictorError {
    fn from(value: io::Error) -> Self {
        AestheticPredictorError::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, AestheticPredictorError>;
