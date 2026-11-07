// Shared TurboQuant constants + WHT for the KV quantize/dequant kernels (ports
// crates/infr-cpu/src/turbo.rs). One thread owns a whole 128-element block. -DTURBO2/3/4 picks the
// centroid table + bit width; the WHT sign tables (extracted verbatim from llama.cpp) are shared.
const float INV_SQRT = 0.088388350;

const int S1[128] = int[128](
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1);
const int S2[128] = int[128](
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1);

#if defined(TURBO2)
const int NBITS = 2;
const int NCENT = 4;
const float CENT[4] = float[4](-0.133462, -0.039994, 0.039994, 0.133462);
const float MID[3] = float[3](-0.086728, 0.0, 0.086728);
#elif defined(TURBO3)
const int NBITS = 3;
const int NCENT = 8;
const float CENT[8] = float[8](-0.190207, -0.118786, -0.066822, -0.021663, 0.021663, 0.066822, 0.118786, 0.190207);
const float MID[7] = float[7](-0.154496, -0.092804, -0.044243, 0.0, 0.044243, 0.092804, 0.154496);
#elif defined(TURBO4)
const int NBITS = 4;
const int NCENT = 16;
const float CENT[16] = float[16](-0.241529,-0.182877,-0.143016,-0.111036,-0.083292,-0.058050,-0.034299,-0.011349,0.011349,0.034299,0.058050,0.083292,0.111036,0.143016,0.182877,0.241529);
const float MID[15] = float[15](-0.212203,-0.162947,-0.127026,-0.097164,-0.070671,-0.046174,-0.022824,0.0,0.022824,0.046174,0.070671,0.097164,0.127026,0.162947,0.212203);
#endif

int nearest_cent(float v) {
    for (int i = 0; i < NCENT - 1; i++) { if (v < MID[i]) return i; }
    return NCENT - 1;
}

// Forward WHT (in place, 128 elems): x*=S1; Hadamard butterfly; x*=inv_sqrt*S2.
void fwht(inout float x[128]) {
    for (int i = 0; i < 128; i++) x[i] *= float(S1[i]);
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    for (int i = 0; i < 128; i++) x[i] *= INV_SQRT * float(S2[i]);
}

// Inverse WHT: swap the sign diagonals.
void fwht_inv(inout float x[128]) {
    for (int i = 0; i < 128; i++) x[i] *= float(S2[i]);
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    for (int i = 0; i < 128; i++) x[i] *= INV_SQRT * float(S1[i]);
}

// Bytes per block for this width: turbo2=34, turbo3=50, turbo4=66.
uint block_bytes() {
#if defined(TURBO2)
    return 34u;
#elif defined(TURBO3)
    return 50u;
#else
    return 66u;
#endif
}
