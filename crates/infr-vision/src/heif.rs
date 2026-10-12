//! Pinned libheif C API; codecs are loaded only for HEIF/AVIF inputs.
use anyhow::{bail, Context, Result};
use image::{DynamicImage, RgbImage};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    path::{Path, PathBuf},
    ptr,
    sync::OnceLock,
};

const MAX_PIXELS: u64 = 64 * 1024 * 1024;
const MAX_SIDE: c_int = 8192;

pub(crate) fn recognizes(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if size < 16 || size > bytes.len() || size > 4096 {
        return false;
    }
    std::iter::once(&bytes[8..12])
        .chain(bytes[16..size].chunks_exact(4))
        .any(|brand| {
            matches!(
                brand,
                b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1" | b"avif" | b"avis"
            )
        })
}

#[repr(C)]
struct Error {
    code: c_int,
    subcode: c_int,
    message: *const c_char,
}
impl Error {
    fn check(self, stage: &str) -> Result<()> {
        if self.code == 0 {
            return Ok(());
        }
        let message = if self.message.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(self.message) }.to_string_lossy()
        };
        bail!(
            "HEIC/AVIF {stage}: {message} (code {}, subcode {})",
            self.code,
            self.subcode
        )
    }
}
type Free = unsafe extern "C" fn(*mut c_void);
// Stable v1 prefix of heif_security_limits; leave all later security limits intact.
#[repr(C)]
struct SecurityLimitsPrefix {
    version: u8,
    max_image_size_pixels: u64,
}
struct Owned {
    pointer: *mut c_void,
    free: Free,
}
impl Drop for Owned {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.free)(self.pointer) };
        }
    }
}

struct Decoder {
    alloc: unsafe extern "C" fn() -> *mut c_void,
    free: Free,
    read: unsafe extern "C" fn(*mut c_void, *const c_void, usize, *const c_void) -> Error,
    threads: unsafe extern "C" fn(*mut c_void, c_int),
    limits: unsafe extern "C" fn(*const c_void) -> *mut SecurityLimitsPrefix,
    primary: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Error,
    handle_free: Free,
    handle_width: unsafe extern "C" fn(*const c_void) -> c_int,
    handle_height: unsafe extern "C" fn(*const c_void) -> c_int,
    decode:
        unsafe extern "C" fn(*const c_void, *mut *mut c_void, c_int, c_int, *const c_void) -> Error,
    image_free: Free,
    width: unsafe extern "C" fn(*const c_void, c_int) -> c_int,
    height: unsafe extern "C" fn(*const c_void, c_int) -> c_int,
    plane: unsafe extern "C" fn(*const c_void, c_int, *mut c_int) -> *const u8,
    // Function pointers and decoder registration remain valid for the process lifetime.
    _libraries: Vec<Library>,
}

fn codec_directory() -> Result<PathBuf> {
    // Explicit override is for source builds and codec tests, never inferred from cwd/PATH.
    if let Some(path) = std::env::var_os("INFR_IMAGE_CODEC_DIR") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            bail!("INFR_IMAGE_CODEC_DIR must be absolute");
        }
        return Ok(path);
    }
    Ok(std::env::current_exe()?
        .parent()
        .context("executable directory missing")?
        .join("image-codecs"))
}

impl Decoder {
    unsafe fn load(directory: &Path) -> Result<Self> {
        let mut libraries = Vec::new();
        #[cfg(target_os = "windows")]
        for name in ["libde265.dll", "dav1d.dll", "heif.dll"] {
            let path = directory.join(name);
            if name == "dav1d.dll" && !path.exists() {
                continue;
            }
            // Restrict dependency lookup to the trusted library directory and Windows system32.
            let library =
                unsafe { libloading::os::windows::Library::load_with_flags(&path, 0x100 | 0x800) }?;
            libraries.push(Library::from(library));
        }
        #[cfg(not(target_os = "windows"))]
        libraries.push(unsafe { Library::new(directory.join("libheif.so.1")) }?);
        let library = libraries.last().context("no image codec library")?;
        macro_rules! symbol {
            ($name:literal) => {
                *unsafe { library.get(concat!($name, "\0").as_bytes()) }?
            };
        }
        let init: unsafe extern "C" fn(*const c_void) -> Error = symbol!("heif_init");
        let decoder = Self {
            alloc: symbol!("heif_context_alloc"),
            free: symbol!("heif_context_free"),
            read: symbol!("heif_context_read_from_memory_without_copy"),
            threads: symbol!("heif_context_set_max_decoding_threads"),
            limits: symbol!("heif_context_get_security_limits"),
            primary: symbol!("heif_context_get_primary_image_handle"),
            handle_free: symbol!("heif_image_handle_release"),
            handle_width: symbol!("heif_image_handle_get_width"),
            handle_height: symbol!("heif_image_handle_get_height"),
            decode: symbol!("heif_decode_image"),
            image_free: symbol!("heif_image_release"),
            width: symbol!("heif_image_get_width"),
            height: symbol!("heif_image_get_height"),
            plane: symbol!("heif_image_get_plane_readonly"),
            _libraries: libraries,
        };
        unsafe { init(ptr::null()) }.check("initialization")?;
        Ok(decoder)
    }

    fn decode(&self, bytes: &[u8]) -> Result<DynamicImage> {
        if bytes.len() > 256 * 1024 * 1024 {
            bail!("HEIC/AVIF compressed image exceeds 256 MiB");
        }
        // Each request owns its context; the borrowed input lives until all handles are released.
        unsafe {
            let context = Owned {
                pointer: (self.alloc)(),
                free: self.free,
            };
            if context.pointer.is_null() {
                bail!("HEIC/AVIF context allocation failed");
            }
            (self.threads)(context.pointer, 2);
            let limits = (self.limits)(context.pointer);
            if limits.is_null() || (*limits).version < 1 {
                bail!("HEIC/AVIF security limits unavailable");
            }
            (*limits).max_image_size_pixels = MAX_PIXELS;
            (self.read)(
                context.pointer,
                bytes.as_ptr().cast(),
                bytes.len(),
                ptr::null(),
            )
            .check("read")?;
            let mut handle = Owned {
                pointer: ptr::null_mut(),
                free: self.handle_free,
            };
            (self.primary)(context.pointer, &mut handle.pointer).check("primary image")?;
            if handle.pointer.is_null() {
                bail!("HEIC/AVIF missing primary image");
            }
            dimensions(
                (self.handle_width)(handle.pointer),
                (self.handle_height)(handle.pointer),
            )?;
            let mut image = Owned {
                pointer: ptr::null_mut(),
                free: self.image_free,
            };
            // RGB/interleaved RGB8. Default options apply rotation, mirroring and crop.
            (self.decode)(handle.pointer, &mut image.pointer, 1, 10, ptr::null())
                .check("decode")?;
            if image.pointer.is_null() {
                bail!("HEIC/AVIF decoder returned no image");
            }
            let (w, h) = dimensions(
                (self.width)(image.pointer, 10),
                (self.height)(image.pointer, 10),
            )?;
            let mut stride = 0;
            let plane = (self.plane)(image.pointer, 10, &mut stride);
            let row = w as usize * 3;
            if plane.is_null() || stride < 0 || (stride as usize) < row {
                bail!("HEIC/AVIF invalid RGB plane");
            }
            let mut pixels = vec![0; row * h as usize];
            for y in 0..h as usize {
                pixels[y * row..(y + 1) * row].copy_from_slice(std::slice::from_raw_parts(
                    plane.add(y * stride as usize),
                    row,
                ));
            }
            Ok(DynamicImage::ImageRgb8(
                RgbImage::from_raw(w, h, pixels).context("invalid RGB dimensions")?,
            ))
        }
    }
}

fn dimensions(w: c_int, h: c_int) -> Result<(u32, u32)> {
    if w <= 0 || h <= 0 || w > MAX_SIDE || h > MAX_SIDE || w as u64 * h as u64 > MAX_PIXELS {
        bail!("HEIC/AVIF image dimensions {w}x{h} exceed the 8192-side / 64-Mpixel limit");
    }
    Ok((w as u32, h as u32))
}

pub(crate) fn decode(bytes: &[u8]) -> Result<DynamicImage> {
    static DECODER: OnceLock<std::result::Result<Decoder, String>> = OnceLock::new();
    let decoder = DECODER.get_or_init(|| {
        codec_directory()
            .and_then(|path| unsafe { Decoder::load(&path) })
            .map_err(|e| format!("{e:#}"))
    });
    match decoder {
        Ok(decoder) => decoder.decode(bytes),
        Err(error) => bail!(
            "HEIC/AVIF decoder unavailable; install the release image-codecs directory: {error}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brands_and_limits() {
        assert!(recognizes(b"\0\0\0\x18ftypheic\0\0\0\0mif1heic"));
        assert!(recognizes(b"\0\0\0\x18ftypavif\0\0\0\0mif1avif"));
        assert!(!recognizes(b"\0\0\0\x18ftypmp42\0\0\0\0mp42isom"));
        assert!(!recognizes(b"\0\0\0\x18ftypheic"));
        for dims in [(0, 1), (-1, 1), (8193, 1), (8192, 8193)] {
            assert!(dimensions(dims.0, dims.1).is_err());
        }
        assert_eq!(dimensions(800, 533).unwrap(), (800, 533));
        assert_eq!(dimensions(8064, 6048).unwrap(), (8064, 6048));
    }

    #[test]
    fn missing_bundle_returns_an_error() {
        let missing =
            std::env::temp_dir().join(format!("infr-missing-codecs-{}", std::process::id()));
        assert!(!missing.exists());
        assert!(unsafe { Decoder::load(&missing) }.is_err());
    }

    #[test]
    #[ignore = "requires the pinned decoder bundle and upstream fixtures"]
    fn bundled_codecs() -> Result<()> {
        use sha2::{Digest, Sha256};
        let root = PathBuf::from(
            std::env::var_os("INFR_IMAGE_CODEC_FIXTURES").context("fixture directory missing")?,
        );
        let cases = [
            (
                "examples/example.heic",
                (1280, 854),
                "e724f687fa08242f0242ecaea6fdf27c285c69acc8be7e7b69f8f1727a6560f2",
            ),
            (
                "examples/example.avif",
                (800, 533),
                "ad9f6900033c9391c25091ab05a788f91e22bf87dccd913f3cb91b3eb1d6a3a1",
            ),
            (
                "fuzzing/data/corpus/colors-with-alpha.heic",
                (64, 64),
                "9a3c0a13b24b1405aaee90867a5c042ceee41750ee72c5a15ae87e19bf095231",
            ),
            (
                "fuzzing/data/corpus/colors-no-alpha.heic",
                (64, 64),
                "9a3c0a13b24b1405aaee90867a5c042ceee41750ee72c5a15ae87e19bf095231",
            ),
        ];
        for (name, dims, hash) in cases {
            let bytes = std::fs::read(root.join(name))?;
            assert!(recognizes(&bytes));
            let image = decode(&bytes)?.to_rgb8();
            assert_eq!(image.dimensions(), dims);
            assert_eq!(format!("{:x}", Sha256::digest(image.as_raw())), hash);
            assert!(decode(&bytes[..bytes.len().min(32)]).is_err());
        }
        for (name, dims) in [
            ("clap_cropped.heic", (64, 64)),
            ("clap_cropped.avif", (64, 64)),
            ("clap_cropped_irot_imir.avif", (64, 64)),
            ("conformance_window_padding.heic", (1, 1)),
            ("rainbow-451x461.heic", (451, 461)),
        ] {
            let image = decode(&std::fs::read(root.join("tests/data").join(name))?)?.to_rgb8();
            assert_eq!(image.dimensions(), dims, "{name}");
        }
        for name in [
            "clap_oversized_ispe_width.avif",
            "clap_oversized_ispe_height.avif",
        ] {
            assert!(
                decode(&std::fs::read(root.join("tests/data").join(name))?).is_err(),
                "{name}"
            );
        }
        // Simultaneous requests use independent decoder contexts.
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| decode(&std::fs::read(root.join(cases[3].0)).unwrap()).unwrap())
                })
                .collect();
            for job in jobs {
                assert_eq!(job.join().unwrap().width(), 64);
            }
        });
        Ok(())
    }
}
