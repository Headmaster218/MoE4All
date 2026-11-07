//! Feature structs for Vulkan extensions that ash 0.38 (headers 1.3.281) predates.
//!
//! Two of the extensions this backend probes — `VK_KHR_shader_bfloat16` and `VK_EXT_shader_float8`
//! — shipped after ash's pinned headers, so ash has no type, no `StructureType` constant, and no
//! `ExtendsPhysicalDeviceFeatures2` impl for them. That is why they were the only two capabilities
//! gated on the extension STRING alone, while every other one here is gated on extension AND
//! feature bit AND the feature actually being enabled on the device.
//!
//! This module supplies the missing halves: the structs are transcribed from the system Vulkan
//! headers (`/usr/include/vulkan/vulkan_core.h`, `VkPhysicalDeviceShaderBfloat16FeaturesKHR` and
//! `VkPhysicalDeviceShaderFloat8FeaturesEXT`), whose layout and `sType` values are fixed ABI, and
//! the chaining is done by hand against the same `pNext` protocol ash's `push_next` implements.
//!
//! **Nothing in here runs on the box it was written on** — an RX 7900 XTX advertises neither
//! extension, so both queries are skipped and both enables are absent. The one thing that IS
//! exercised on this hardware is the raw-chaining MECHANISM: [`tests::raw_chaining_matches_ash`]
//! hand-rolls a struct ash DOES have (`VkPhysicalDeviceShaderFloat16Int8Features`) and asserts the
//! bits come back identical to ash's typed query.

use std::ffi::c_void;
use std::ptr;

use ash::vk;

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR` (vulkan_core.h).
const ST_SHADER_BFLOAT16_FEATURES_KHR: i32 = 1_000_141_000;
/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT` (vulkan_core.h).
const ST_SHADER_FLOAT8_FEATURES_EXT: i32 = 1_000_567_000;

/// `VkPhysicalDeviceShaderBfloat16FeaturesKHR`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub shader_bfloat16_type: vk::Bool32,
    pub shader_bfloat16_dot_product: vk::Bool32,
    pub shader_bfloat16_cooperative_matrix: vk::Bool32,
}

/// `VkPhysicalDeviceShaderFloat8FeaturesEXT`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ShaderFloat8Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub shader_float8: vk::Bool32,
    pub shader_float8_cooperative_matrix: vk::Bool32,
}

impl Default for ShaderBfloat16Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(ST_SHADER_BFLOAT16_FEATURES_KHR),
            p_next: ptr::null_mut(),
            shader_bfloat16_type: 0,
            shader_bfloat16_dot_product: 0,
            shader_bfloat16_cooperative_matrix: 0,
        }
    }
}

impl Default for ShaderFloat8Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(ST_SHADER_FLOAT8_FEATURES_EXT),
            p_next: ptr::null_mut(),
            shader_float8: 0,
            shader_float8_cooperative_matrix: 0,
        }
    }
}

/// What the device reports for the two post-ash extensions. Every field is false when the matching
/// extension is absent — the query is not even issued then (chaining a struct the driver does not
/// know is undefined behaviour).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PostAshFeatures {
    /// `shaderBFloat16Type` — bf16 as a scalar/vector shader type.
    pub bf16_type: bool,
    /// `shaderBFloat16CooperativeMatrix` — bf16 as a cooperative-matrix COMPONENT type, which is
    /// what `native_gemm_warp.comp`'s `-DBF16CM` build declares (`GL_EXT_bfloat16`, operand type
    /// `bfloat16_t`). Enumerating a bf16 coopmat CONFIG is a separate, weaker statement.
    pub bf16_coopmat: bool,
    /// `shaderFloat8` — E4M3/E5M2 as a shader type (`GL_EXT_float_e4m3`'s `floate4m3_t`).
    pub f8: bool,
    /// `shaderFloat8CooperativeMatrix` — fp8 as a cooperative-matrix component type.
    pub f8_coopmat: bool,
}

/// Query the two post-ash feature structs, skipping either whose extension the device does not
/// advertise.
///
/// # Safety
/// `physical_device` must belong to `instance`. The `want_*` flags MUST be the result of an
/// extension-presence check: chaining a struct an implementation does not recognise is undefined
/// behaviour, not a graceful no-op.
pub(crate) unsafe fn query_post_ash_features(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    want_bf16: bool,
    want_f8: bool,
) -> PostAshFeatures {
    if !(want_bf16 || want_f8) {
        return PostAshFeatures::default();
    }
    let mut bf16 = ShaderBfloat16Features::default();
    let mut f8 = ShaderFloat8Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default();
    // Hand-built pNext chain, newest first — the same prepend order ash's `push_next` uses.
    let mut head: *mut c_void = ptr::null_mut();
    if want_bf16 {
        bf16.p_next = head;
        head = std::ptr::from_mut(&mut bf16).cast();
    }
    if want_f8 {
        f8.p_next = head;
        head = std::ptr::from_mut(&mut f8).cast();
    }
    features2.p_next = head;
    instance.get_physical_device_features2(physical_device, &mut features2);
    PostAshFeatures {
        bf16_type: want_bf16 && bf16.shader_bfloat16_type != 0,
        bf16_coopmat: want_bf16 && bf16.shader_bfloat16_cooperative_matrix != 0,
        f8: want_f8 && f8.shader_float8 != 0,
        f8_coopmat: want_f8 && f8.shader_float8_cooperative_matrix != 0,
    }
}

impl ShaderBfloat16Features {
    /// The struct to CHAIN INTO `VkDeviceCreateInfo` to enable exactly the bits `f` reported (a
    /// device create that asks for a feature the device reports false FAILS, so this never asks for
    /// more than was found).
    pub(crate) fn enable(f: &PostAshFeatures) -> Self {
        Self {
            shader_bfloat16_type: f.bf16_type as vk::Bool32,
            shader_bfloat16_cooperative_matrix: f.bf16_coopmat as vk::Bool32,
            ..Self::default()
        }
    }
}

impl ShaderFloat8Features {
    /// See [`ShaderBfloat16Features::enable`].
    pub(crate) fn enable(f: &PostAshFeatures) -> Self {
        Self {
            shader_float8: f.f8 as vk::Bool32,
            shader_float8_cooperative_matrix: f.f8_coopmat as vk::Bool32,
            ..Self::default()
        }
    }
}

/// A `#[repr(C)]` Vulkan feature struct this module defines: it owns an `sType` set at construction
/// and a `pNext` slot the chaining helper writes. Implemented by naming the FIELD (no offset
/// arithmetic), so the chain cannot go wrong if a struct's layout ever changes.
pub(crate) trait RawFeatureStruct: Sized {
    fn p_next_slot(&mut self) -> &mut *mut c_void;
}

impl RawFeatureStruct for ShaderBfloat16Features {
    fn p_next_slot(&mut self) -> &mut *mut c_void {
        &mut self.p_next
    }
}

impl RawFeatureStruct for ShaderFloat8Features {
    fn p_next_slot(&mut self) -> &mut *mut c_void {
        &mut self.p_next
    }
}

/// Prepend a raw feature struct to `ci`'s `pNext` chain — the hand-rolled `push_next` for structs
/// ash has no `ExtendsDeviceCreateInfo` impl for. Every struct ash already pushed stays reachable
/// behind this one.
///
/// # Safety
/// `raw` must outlive the `vkCreateDevice` call that reads `ci`, and the extension defining it must
/// be in `ci`'s enabled-extension list.
pub(crate) unsafe fn chain_into_device_ci<T: RawFeatureStruct>(
    ci: &mut vk::DeviceCreateInfo<'_>,
    raw: &mut T,
) {
    *raw.p_next_slot() = ci.p_next as *mut c_void;
    ci.p_next = std::ptr::from_mut(raw).cast::<c_void>().cast_const();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MECHANISM check: hand-roll a feature struct ash DOES have a type for, chain it the same
    /// way [`query_post_ash_features`] chains the two it does not, and require the driver to fill
    /// it identically to ash's typed query.
    ///
    /// This is what makes the bf16/f8 code above more than an untested guess on this box: the
    /// hardware here advertises neither extension, so their queries never run, but the raw sType +
    /// pNext protocol they depend on is exercised end-to-end against a real driver here. The
    /// shader-float16-int8 features struct is used because RDNA3 reports BOTH its bits true, so a
    /// chain that silently did nothing would leave them false and fail.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn raw_chaining_matches_ash() {
        #[repr(C)]
        struct RawFloat16Int8 {
            s_type: vk::StructureType,
            p_next: *mut c_void,
            shader_float16: vk::Bool32,
            shader_int8: vk::Bool32,
        }
        let be = crate::VulkanBackend::new().expect("vulkan backend");
        let (instance, pd) = (&be.shared.instance, be.shared.physical_device);

        let mut typed = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut typed);
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };

        let mut raw = RawFloat16Int8 {
            s_type: vk::StructureType::from_raw(1_000_082_000),
            p_next: ptr::null_mut(),
            shader_float16: 0,
            shader_int8: 0,
        };
        let mut f2_raw = vk::PhysicalDeviceFeatures2 {
            p_next: std::ptr::from_mut(&mut raw).cast(),
            ..Default::default()
        };
        unsafe { instance.get_physical_device_features2(pd, &mut f2_raw) };

        assert_eq!(
            (raw.shader_float16 != 0, raw.shader_int8 != 0),
            (typed.shader_float16 != 0, typed.shader_int8 != 0),
            "a hand-chained feature struct must read back what ash's typed one does"
        );
        assert!(
            raw.shader_int8 != 0,
            "this device reports shaderInt8, so a chain that did nothing would be caught here"
        );
    }
}
