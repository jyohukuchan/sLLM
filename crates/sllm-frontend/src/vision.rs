//! Bounded Qwen3.5 image input and deterministic Qwen2-VL-compatible patching.

use base64::Engine as _;
use image::codecs::gif::GifDecoder;
use image::imageops::FilterType;
use image::{AnimationDecoder, ImageFormat, ImageReader, RgbImage};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

pub const MAX_ENCODED_IMAGE_BYTES_V1: usize = 32 * 1024 * 1024;
pub const MIN_IMAGE_PIXELS_V1: u64 = 65_536;
pub const MAX_IMAGE_PIXELS_V1: u64 = 16_777_216;
pub const QWEN35_PATCH_SIZE: u32 = 16;
pub const QWEN35_TEMPORAL_PATCH_SIZE: u32 = 2;
pub const QWEN35_MERGE_SIZE: u32 = 2;
pub const MAX_IMAGE_COUNT_V1: usize = 2;
pub const MAX_TOTAL_VISUAL_TOKENS_V1: u64 = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionImageFormatV1 {
    Png,
    Jpeg,
    WebP,
    Gif,
}

impl VisionImageFormatV1 {
    fn image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::WebP => ImageFormat::WebP,
            Self::Gif => ImageFormat::Gif,
        }
    }

    fn mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::WebP => "image/webp",
            Self::Gif => "image/gif",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedImageBytesV1 {
    bytes: Vec<u8>,
    format: VisionImageFormatV1,
    sha256: [u8; 32],
}

impl BoundedImageBytesV1 {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, VisionErrorV1> {
        if bytes.is_empty() || bytes.len() > MAX_ENCODED_IMAGE_BYTES_V1 {
            return Err(VisionErrorV1::EncodedSize);
        }
        let format = detect_format(&bytes)?;
        let sha256 = Sha256::digest(&bytes).into();
        Ok(Self {
            bytes,
            format,
            sha256,
        })
    }

    pub fn from_data_url(value: &str) -> Result<Self, VisionErrorV1> {
        let (header, payload) = value.split_once(',').ok_or(VisionErrorV1::InvalidDataUrl)?;
        let mime = header
            .strip_prefix("data:")
            .and_then(|header| header.strip_suffix(";base64"))
            .ok_or(VisionErrorV1::InvalidDataUrl)?;
        if payload.len() > MAX_ENCODED_IMAGE_BYTES_V1.saturating_mul(4) / 3 + 4 {
            return Err(VisionErrorV1::EncodedSize);
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|_| VisionErrorV1::InvalidDataUrl)?;
        let image = Self::from_bytes(bytes)?;
        if mime != image.format.mime() {
            return Err(VisionErrorV1::MimeMagicMismatch);
        }
        Ok(image)
    }

    pub fn from_local_path(path: impl AsRef<Path>) -> Result<Self, VisionErrorV1> {
        let file = File::open(path).map_err(|_| VisionErrorV1::Read)?;
        let length = file.metadata().map_err(|_| VisionErrorV1::Read)?.len();
        if length == 0 || length > MAX_ENCODED_IMAGE_BYTES_V1 as u64 {
            return Err(VisionErrorV1::EncodedSize);
        }
        let mut bytes = Vec::with_capacity(length as usize);
        file.take(MAX_ENCODED_IMAGE_BYTES_V1 as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| VisionErrorV1::Read)?;
        Self::from_bytes(bytes)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn format(&self) -> VisionImageFormatV1 {
        self.format
    }

    pub fn digest_hex(&self) -> String {
        hex(&self.sha256)
    }

    pub fn decode_rgb(&self) -> Result<DecodedRgbImageV1, VisionErrorV1> {
        if self.format == VisionImageFormatV1::Gif {
            let decoder = GifDecoder::new(Cursor::new(&self.bytes))
                .map_err(|_| VisionErrorV1::MalformedImage)?;
            let mut frames = decoder.into_frames();
            if frames
                .next()
                .transpose()
                .map_err(|_| VisionErrorV1::MalformedImage)?
                .is_none()
            {
                return Err(VisionErrorV1::MalformedImage);
            }
            if frames
                .next()
                .transpose()
                .map_err(|_| VisionErrorV1::MalformedImage)?
                .is_some()
            {
                return Err(VisionErrorV1::AnimatedImage);
            }
        }
        let reader = ImageReader::with_format(Cursor::new(&self.bytes), self.format.image_format());
        let (width, height) = reader
            .into_dimensions()
            .map_err(|_| VisionErrorV1::MalformedImage)?;
        validate_dimensions(width, height)?;
        let image = image::load_from_memory_with_format(&self.bytes, self.format.image_format())
            .map_err(|_| VisionErrorV1::MalformedImage)?
            .to_rgb8();
        if image.width() != width || image.height() != height {
            return Err(VisionErrorV1::MalformedImage);
        }
        Ok(DecodedRgbImageV1 {
            width,
            height,
            pixels: image.into_raw(),
            source_sha256: self.sha256,
            format: self.format,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedRgbImageV1 {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    source_sha256: [u8; 32],
    format: VisionImageFormatV1,
}

impl DecodedRgbImageV1 {
    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub const fn format(&self) -> VisionImageFormatV1 {
        self.format
    }

    pub fn source_digest_hex(&self) -> String {
        hex(&self.source_sha256)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionPatchPositionV1 {
    pub temporal: u32,
    pub row: u32,
    pub column: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedVisionInputV1 {
    pub resized_width: u32,
    pub resized_height: u32,
    pub grid_thw: [u32; 3],
    pub patch_width: usize,
    pub patches: Vec<f32>,
    pub merged_positions: Vec<VisionPatchPositionV1>,
    pub visual_tokens: u64,
    pub patch_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
}

impl ProcessedVisionInputV1 {
    pub fn patch_digest_hex(&self) -> String {
        hex(&self.patch_sha256)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen35VisionProcessorV1;

impl Qwen35VisionProcessorV1 {
    pub fn process(
        &self,
        image: &DecodedRgbImageV1,
    ) -> Result<ProcessedVisionInputV1, VisionErrorV1> {
        validate_dimensions(image.width, image.height)?;
        let (resized_height, resized_width) = smart_resize(image.height, image.width)?;
        let source = RgbImage::from_raw(image.width, image.height, image.pixels.clone())
            .ok_or(VisionErrorV1::MalformedImage)?;
        // EXIF orientation is intentionally ignored. Pixel order is the
        // decoder's canonical raster order and is independent of metadata.
        let resized = image::imageops::resize(
            &source,
            resized_width,
            resized_height,
            FilterType::CatmullRom,
        );
        let grid_t = 1_u32;
        let grid_h = resized_height / QWEN35_PATCH_SIZE;
        let grid_w = resized_width / QWEN35_PATCH_SIZE;
        if grid_h % QWEN35_MERGE_SIZE != 0 || grid_w % QWEN35_MERGE_SIZE != 0 {
            return Err(VisionErrorV1::PatchGeometry);
        }
        let patch_rows = u64::from(grid_h)
            .checked_mul(u64::from(grid_w))
            .ok_or(VisionErrorV1::Overflow)?;
        let patch_width = 3_usize
            .checked_mul(QWEN35_TEMPORAL_PATCH_SIZE as usize)
            .and_then(|value| value.checked_mul(QWEN35_PATCH_SIZE as usize))
            .and_then(|value| value.checked_mul(QWEN35_PATCH_SIZE as usize))
            .ok_or(VisionErrorV1::Overflow)?;
        let patch_values = usize::try_from(patch_rows)
            .ok()
            .and_then(|rows| rows.checked_mul(patch_width))
            .ok_or(VisionErrorV1::Overflow)?;
        let mut patches = Vec::with_capacity(patch_values);
        let mut merged_positions = Vec::with_capacity(
            usize::try_from(patch_rows / u64::from(QWEN35_MERGE_SIZE.pow(2)))
                .map_err(|_| VisionErrorV1::Overflow)?,
        );
        for block_row in 0..grid_h / QWEN35_MERGE_SIZE {
            for block_column in 0..grid_w / QWEN35_MERGE_SIZE {
                merged_positions.push(VisionPatchPositionV1 {
                    temporal: 0,
                    row: block_row,
                    column: block_column,
                });
                for merge_row in 0..QWEN35_MERGE_SIZE {
                    for merge_column in 0..QWEN35_MERGE_SIZE {
                        for channel in 0..3 {
                            for _temporal in 0..QWEN35_TEMPORAL_PATCH_SIZE {
                                for patch_row in 0..QWEN35_PATCH_SIZE {
                                    for patch_column in 0..QWEN35_PATCH_SIZE {
                                        let y = (block_row * QWEN35_MERGE_SIZE + merge_row)
                                            * QWEN35_PATCH_SIZE
                                            + patch_row;
                                        let x = (block_column * QWEN35_MERGE_SIZE + merge_column)
                                            * QWEN35_PATCH_SIZE
                                            + patch_column;
                                        let value = resized.get_pixel(x, y).0[channel];
                                        patches.push((f32::from(value) / 255.0 - 0.5) / 0.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if patches.len() != patch_values {
            return Err(VisionErrorV1::PatchGeometry);
        }
        let visual_tokens = patch_rows / u64::from(QWEN35_MERGE_SIZE.pow(2));
        if visual_tokens == 0 || visual_tokens > MAX_TOTAL_VISUAL_TOKENS_V1 {
            return Err(VisionErrorV1::VisualTokenLimit);
        }
        let mut hash = Sha256::new();
        for value in &patches {
            hash.update(value.to_le_bytes());
        }
        Ok(ProcessedVisionInputV1 {
            resized_width,
            resized_height,
            grid_thw: [grid_t, grid_h, grid_w],
            patch_width,
            patches,
            merged_positions,
            visual_tokens,
            patch_sha256: hash.finalize().into(),
            source_sha256: image.source_sha256,
        })
    }

    pub fn process_many(
        &self,
        images: &[DecodedRgbImageV1],
    ) -> Result<Vec<ProcessedVisionInputV1>, VisionErrorV1> {
        if images.is_empty() || images.len() > MAX_IMAGE_COUNT_V1 {
            return Err(VisionErrorV1::ImageCount);
        }
        let processed = images
            .iter()
            .map(|image| self.process(image))
            .collect::<Result<Vec<_>, _>>()?;
        let total = processed.iter().try_fold(0_u64, |total, image| {
            total
                .checked_add(image.visual_tokens)
                .ok_or(VisionErrorV1::Overflow)
        })?;
        if total > MAX_TOTAL_VISUAL_TOKENS_V1 {
            return Err(VisionErrorV1::VisualTokenLimit);
        }
        Ok(processed)
    }
}

fn detect_format(bytes: &[u8]) -> Result<VisionImageFormatV1, VisionErrorV1> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Ok(VisionImageFormatV1::Png)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Ok(VisionImageFormatV1::Jpeg)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Ok(VisionImageFormatV1::Gif)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Ok(VisionImageFormatV1::WebP)
    } else {
        Err(VisionErrorV1::UnsupportedFormat)
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), VisionErrorV1> {
    if width == 0 || height == 0 {
        return Err(VisionErrorV1::PixelArea);
    }
    let area = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(VisionErrorV1::Overflow)?;
    if !(MIN_IMAGE_PIXELS_V1..=MAX_IMAGE_PIXELS_V1).contains(&area) {
        return Err(VisionErrorV1::PixelArea);
    }
    let long = width.max(height) as f64;
    let short = width.min(height) as f64;
    if long / short > 200.0 {
        return Err(VisionErrorV1::AspectRatio);
    }
    Ok(())
}

fn smart_resize(height: u32, width: u32) -> Result<(u32, u32), VisionErrorV1> {
    validate_dimensions(width, height)?;
    let factor = f64::from(QWEN35_PATCH_SIZE * QWEN35_MERGE_SIZE);
    let mut resized_height = (f64::from(height) / factor).round_ties_even() * factor;
    let mut resized_width = (f64::from(width) / factor).round_ties_even() * factor;
    if resized_height * resized_width > MAX_IMAGE_PIXELS_V1 as f64 {
        let beta = (f64::from(height) * f64::from(width) / MAX_IMAGE_PIXELS_V1 as f64).sqrt();
        resized_height = ((f64::from(height) / beta / factor).floor() * factor).max(factor);
        resized_width = ((f64::from(width) / beta / factor).floor() * factor).max(factor);
    } else if resized_height * resized_width < MIN_IMAGE_PIXELS_V1 as f64 {
        let beta = (MIN_IMAGE_PIXELS_V1 as f64 / (f64::from(height) * f64::from(width))).sqrt();
        resized_height = (f64::from(height) * beta / factor).ceil() * factor;
        resized_width = (f64::from(width) * beta / factor).ceil() * factor;
    }
    let resized_height =
        u32::try_from(resized_height as u64).map_err(|_| VisionErrorV1::Overflow)?;
    let resized_width = u32::try_from(resized_width as u64).map_err(|_| VisionErrorV1::Overflow)?;
    Ok((resized_height, resized_width))
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisionErrorV1 {
    Read,
    EncodedSize,
    InvalidDataUrl,
    MimeMagicMismatch,
    UnsupportedFormat,
    MalformedImage,
    AnimatedImage,
    PixelArea,
    AspectRatio,
    PatchGeometry,
    VisualTokenLimit,
    ImageCount,
    Overflow,
}

impl fmt::Display for VisionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Read => "image bytes could not be read",
            Self::EncodedSize => "encoded image exceeds the bounded size",
            Self::InvalidDataUrl => "image URL is not a canonical base64 data URL",
            Self::MimeMagicMismatch => "image MIME type differs from magic bytes",
            Self::UnsupportedFormat => "image format is unsupported",
            Self::MalformedImage => "image is malformed",
            Self::AnimatedImage => "animated images are unsupported",
            Self::PixelArea => "decoded image pixel area is outside the supported range",
            Self::AspectRatio => "image aspect ratio exceeds 200",
            Self::PatchGeometry => "image patch geometry is inconsistent",
            Self::VisualTokenLimit => "visual token count exceeds the request limit",
            Self::ImageCount => "request image count is outside 1..=2",
            Self::Overflow => "image size arithmetic overflowed",
        })
    }
}

impl std::error::Error for VisionErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
        });
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    #[test]
    fn magic_data_url_and_pixel_boundaries_fail_closed() {
        let bytes = png(256, 256);
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        let image = BoundedImageBytesV1::from_data_url(&url).unwrap();
        assert_eq!(image.format(), VisionImageFormatV1::Png);
        assert_eq!(image.decode_rgb().unwrap().pixels().len(), 256 * 256 * 3);

        let wrong = url.replacen("image/png", "image/jpeg", 1);
        assert_eq!(
            BoundedImageBytesV1::from_data_url(&wrong).unwrap_err(),
            VisionErrorV1::MimeMagicMismatch
        );
        assert_eq!(
            BoundedImageBytesV1::from_bytes(png(255, 257))
                .unwrap()
                .decode_rgb()
                .unwrap_err(),
            VisionErrorV1::PixelArea
        );
    }

    #[test]
    fn patch_order_shape_normalization_and_digest_are_deterministic() {
        let source = BoundedImageBytesV1::from_bytes(png(257, 257))
            .unwrap()
            .decode_rgb()
            .unwrap();
        let processor = Qwen35VisionProcessorV1;
        let first = processor.process(&source).unwrap();
        let second = processor.process(&source).unwrap();
        assert_eq!(first.grid_thw, [1, 16, 16]);
        assert_eq!(first.patch_width, 1_536);
        assert_eq!(first.patches.len(), 16 * 16 * 1_536);
        assert_eq!(first.visual_tokens, 64);
        assert_eq!(first.merged_positions.len(), 64);
        assert_eq!(first.patch_sha256, second.patch_sha256);
        assert!(
            first
                .patches
                .iter()
                .all(|value| (-1.0..=1.0).contains(value))
        );
    }

    #[test]
    fn patch_digest_matches_independent_numpy_oracle() {
        let source = BoundedImageBytesV1::from_bytes(png(256, 256))
            .unwrap()
            .decode_rgb()
            .unwrap();
        let output = Qwen35VisionProcessorV1.process(&source).unwrap();
        assert_eq!(
            output.patch_digest_hex(),
            "f1e51663a9ea2832a67e5157ca11bc42206aaf186897866dab8c779d08ee3a2e"
        );
    }

    #[test]
    fn size_and_count_boundaries_include_both_sides() {
        assert!(validate_dimensions(255, 257).is_err());
        assert!(validate_dimensions(256, 256).is_ok());
        // The exact +1 area is prime and is rejected by the independent
        // aspect-ratio bound after passing the area bound.
        assert_eq!(
            validate_dimensions(1, 65_537),
            Err(VisionErrorV1::AspectRatio)
        );
        assert!(validate_dimensions(4_095, 4_097).is_ok());
        assert!(validate_dimensions(4_096, 4_096).is_ok());
        assert!(validate_dimensions(1, 16_777_217).is_err());
        assert!(Qwen35VisionProcessorV1.process_many(&[]).is_err());
    }
}
