/**
 * Fused GPU preprocessing kernel for pose estimation inference.
 *
 * Single kernel launch performs:
 *   1. NV12 → RGB colorspace conversion (BT.709)
 *   2. Bilinear resize to model input dimensions
 *   3. Normalization: (pixel/255 - mean) / std  (ImageNet or ZeroOne)
 *   4. Layout conversion to NCHW float32
 *
 * Input:  NV12 frame in VRAM (from VA-API decode, zero-copy via DMA-BUF)
 * Output: float32 tensor [1, 3, dst_h, dst_w] in NCHW layout, ready for ORT
 *
 * Also supports RGB24 input (for fallback path where VA-API → CPU → hipMemcpy).
 */

#include <hip/hip_runtime.h>
#include <cstdint>

// ── BT.709 YUV → RGB conversion ────────────────────────────────────

__device__ __forceinline__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ void yuv_to_rgb_bt709(uint8_t y_val, uint8_t u_val, uint8_t v_val,
                                  float &r, float &g, float &b) {
    // BT.709 limited range → [0,1]
    float y = (y_val - 16.0f) / 219.0f;
    float u = (u_val - 128.0f) / 224.0f;
    float v = (v_val - 128.0f) / 224.0f;

    r = clamp01(y + 1.5748f * v);
    g = clamp01(y - 0.1873f * u - 0.4681f * v);
    b = clamp01(y + 1.8556f * u);
}

// ── Bilinear sampling helpers ───────────────────────────────────────

__device__ __forceinline__ uint8_t sample_plane(
    const uint8_t *plane, int x, int y, int pitch, int w, int h
) {
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    return plane[y * pitch + x];
}

__device__ void bilinear_sample_nv12(
    const uint8_t *y_plane, int y_pitch,
    const uint8_t *uv_plane, int uv_pitch,
    int src_w, int src_h,
    float sx, float sy,
    float &r, float &g, float &b
) {
    // Bilinear interpolation on the Y plane
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = sx - x0;
    float fy = sy - y0;

    float y00 = sample_plane(y_plane, x0, y0, y_pitch, src_w, src_h);
    float y10 = sample_plane(y_plane, x1, y0, y_pitch, src_w, src_h);
    float y01 = sample_plane(y_plane, x0, y1, y_pitch, src_w, src_h);
    float y11 = sample_plane(y_plane, x1, y1, y_pitch, src_w, src_h);

    float y_val = y00 * (1.0f - fx) * (1.0f - fy)
                + y10 * fx * (1.0f - fy)
                + y01 * (1.0f - fx) * fy
                + y11 * fx * fy;

    // UV plane is half resolution, interleaved U,V pairs
    float uv_sx = sx * 0.5f;
    float uv_sy = sy * 0.5f;
    int uv_x0 = (int)floorf(uv_sx);
    int uv_y0 = (int)floorf(uv_sy);
    int uv_x1 = uv_x0 + 1;
    int uv_y1 = uv_y0 + 1;
    float uv_fx = uv_sx - uv_x0;
    float uv_fy = uv_sy - uv_y0;

    int uv_w = src_w / 2;
    int uv_h = src_h / 2;

    auto sample_uv = [&](int ux, int uy, uint8_t &u, uint8_t &v) {
        ux = min(max(ux, 0), uv_w - 1);
        uy = min(max(uy, 0), uv_h - 1);
        u = uv_plane[uy * uv_pitch + ux * 2];
        v = uv_plane[uy * uv_pitch + ux * 2 + 1];
    };

    uint8_t u00, v00, u10, v10, u01, v01, u11, v11;
    sample_uv(uv_x0, uv_y0, u00, v00);
    sample_uv(uv_x1, uv_y0, u10, v10);
    sample_uv(uv_x0, uv_y1, u01, v01);
    sample_uv(uv_x1, uv_y1, u11, v11);

    float u_val = u00 * (1.0f - uv_fx) * (1.0f - uv_fy)
                + u10 * uv_fx * (1.0f - uv_fy)
                + u01 * (1.0f - uv_fx) * uv_fy
                + u11 * uv_fx * uv_fy;

    float v_val = v00 * (1.0f - uv_fx) * (1.0f - uv_fy)
                + v10 * uv_fx * (1.0f - uv_fy)
                + v01 * (1.0f - uv_fx) * uv_fy
                + v11 * uv_fx * uv_fy;

    yuv_to_rgb_bt709((uint8_t)y_val, (uint8_t)u_val, (uint8_t)v_val, r, g, b);
}

// ── Kernel: NV12 → normalized NCHW float32 ──────────────────────────

/**
 * @param y_plane     Pointer to Y plane in VRAM (from VA-API DMA-BUF import)
 * @param uv_plane    Pointer to UV plane (y_plane + y_pitch * src_h, or separate offset)
 * @param y_pitch     Row pitch of Y plane in bytes
 * @param uv_pitch    Row pitch of UV plane in bytes
 * @param src_w       Source frame width
 * @param src_h       Source frame height
 * @param output      Output float32 tensor [C, dst_h, dst_w] in NCHW layout
 * @param dst_w       Model input width
 * @param dst_h       Model input height
 * @param mean_r/g/b  Normalization mean per channel (0 for ZeroOne)
 * @param std_r/g/b   Normalization std per channel (1 for ZeroOne)
 */
extern "C" __global__ void preprocess_nv12_to_nchw(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    int y_pitch, int uv_pitch,
    int src_w, int src_h,
    float *output,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // Map output pixel to source coordinates
    float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;

    float r, g, b;
    bilinear_sample_nv12(y_plane, y_pitch, uv_plane, uv_pitch,
                         src_w, src_h, sx, sy, r, g, b);

    // Normalize: r,g,b are already in [0,1] from yuv_to_rgb_bt709
    r = (r - mean_r) / std_r;
    g = (g - mean_g) / std_g;
    b = (b - mean_b) / std_b;

    // Write NCHW layout: output[c * dst_h * dst_w + y * dst_w + x]
    int plane_size = dst_h * dst_w;
    int idx = y * dst_w + x;
    output[0 * plane_size + idx] = r;
    output[1 * plane_size + idx] = g;
    output[2 * plane_size + idx] = b;
}

// ── Kernel: RGB24 → normalized NCHW float32 (fallback path) ────────

/**
 * For the fallback path: CPU-decoded RGB24 data uploaded to GPU via hipMemcpy.
 *
 * @param rgb_data    Pointer to packed RGB24 data [src_h, src_w, 3] in VRAM
 * @param src_w       Source frame width
 * @param src_h       Source frame height
 * @param src_pitch   Row pitch in bytes (may include padding)
 * @param output      Output float32 tensor [C, dst_h, dst_w] in NCHW layout
 * @param dst_w       Model input width
 * @param dst_h       Model input height
 * @param mean_r/g/b  Normalization mean per channel
 * @param std_r/g/b   Normalization std per channel
 */
extern "C" __global__ void preprocess_rgb_to_nchw(
    const uint8_t *rgb_data,
    int src_w, int src_h, int src_pitch,
    float *output,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // Map output pixel to source coordinates
    float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;

    // Bilinear interpolation on RGB
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);

    auto sample = [&](int px, int py, int ch) -> float {
        return (float)rgb_data[py * src_pitch + px * 3 + ch];
    };

    float inv255 = 1.0f / 255.0f;
    float r = (sample(x0, y0, 0) * (1-fx)*(1-fy) + sample(x1, y0, 0) * fx*(1-fy)
             + sample(x0, y1, 0) * (1-fx)*fy     + sample(x1, y1, 0) * fx*fy) * inv255;
    float g = (sample(x0, y0, 1) * (1-fx)*(1-fy) + sample(x1, y0, 1) * fx*(1-fy)
             + sample(x0, y1, 1) * (1-fx)*fy     + sample(x1, y1, 1) * fx*fy) * inv255;
    float b = (sample(x0, y0, 2) * (1-fx)*(1-fy) + sample(x1, y0, 2) * fx*(1-fy)
             + sample(x0, y1, 2) * (1-fx)*fy     + sample(x1, y1, 2) * fx*fy) * inv255;

    r = (r - mean_r) / std_r;
    g = (g - mean_g) / std_g;
    b = (b - mean_b) / std_b;

    int plane_size = dst_h * dst_w;
    int idx = y * dst_w + x;
    output[0 * plane_size + idx] = r;
    output[1 * plane_size + idx] = g;
    output[2 * plane_size + idx] = b;
}

// ── Batched variant: preprocess multiple frames in a single launch ──

/**
 * Batch NV12 → NCHW. Each frame is a separate NV12 surface.
 *
 * @param y_planes    Array of N Y-plane device pointers
 * @param uv_planes   Array of N UV-plane device pointers
 * @param y_pitches   Array of N Y-plane pitches
 * @param uv_pitches  Array of N UV-plane pitches
 * @param src_ws      Array of N source widths
 * @param src_hs      Array of N source heights
 * @param output      Output tensor [N, 3, dst_h, dst_w]
 * @param dst_w       Model input width (same for all frames)
 * @param dst_h       Model input height (same for all frames)
 * @param mean/std    Normalization params
 */
extern "C" __global__ void preprocess_nv12_batch_to_nchw(
    const uint8_t **y_planes,
    const uint8_t **uv_planes,
    const int *y_pitches, const int *uv_pitches,
    const int *src_ws, const int *src_hs,
    float *output,
    int dst_w, int dst_h,
    int batch_size,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;  // batch index

    if (x >= dst_w || y >= dst_h || n >= batch_size) return;

    int src_w = src_ws[n];
    int src_h = src_hs[n];
    float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;

    float r, g, b;
    bilinear_sample_nv12(y_planes[n], y_pitches[n], uv_planes[n], uv_pitches[n],
                         src_w, src_h, sx, sy, r, g, b);

    r = (r - mean_r) / std_r;
    g = (g - mean_g) / std_g;
    b = (b - mean_b) / std_b;

    int plane_size = dst_h * dst_w;
    int frame_size = 3 * plane_size;
    int idx = y * dst_w + x;
    float *frame_out = output + n * frame_size;
    frame_out[0 * plane_size + idx] = r;
    frame_out[1 * plane_size + idx] = g;
    frame_out[2 * plane_size + idx] = b;
}
