#ifndef MANDREL_VORTEX_H
#define MANDREL_VORTEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mandrel_vortex_backend mandrel_vortex_backend_t;

enum {
    MANDREL_VORTEX_STATUS_OK = 0,
    MANDREL_VORTEX_STATUS_NULL_POINTER = 1,
    MANDREL_VORTEX_STATUS_INVALID_ARGUMENT = 2,
    MANDREL_VORTEX_STATUS_RUNTIME_ERROR = 3,
    MANDREL_VORTEX_STATUS_PANIC = 255,
};

enum {
    MANDREL_VORTEX_DTYPE_I8 = 1,
    MANDREL_VORTEX_DTYPE_I32 = 2,
};

typedef struct mandrel_vortex_backend_config {
    size_t struct_size;
    const char *runtime_library_path; /* optional; NULL uses Rust-side runtime search */
    const char *kernel_vxbin_path;    /* required: matmul_i8_i32 kernel.vxbin */
    uint32_t device_index;            /* usually 0 */
    uint32_t flags;                   /* reserved, must be 0 */
} mandrel_vortex_backend_config_t;

typedef struct mandrel_vortex_mul_mat_desc {
    size_t struct_size;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lhs_stride; /* 0 means k; current ABI supports contiguous row-major only */
    uint32_t rhs_stride; /* 0 means n; current ABI supports contiguous row-major only */
    uint32_t out_stride; /* 0 means n; current ABI supports contiguous row-major only */
    uint32_t lhs_dtype;
    uint32_t rhs_dtype;
    uint32_t out_dtype;
    uint32_t flags; /* reserved, must be 0 */
} mandrel_vortex_mul_mat_desc_t;

const char *mandrel_vortex_backend_name(void);
const char *mandrel_vortex_status_message(int32_t status);

/* Thread-local; valid until the next mandrel_vortex_* call on the same thread. */
const char *mandrel_vortex_last_error_message(void);

int32_t mandrel_vortex_backend_create(
    const mandrel_vortex_backend_config_t *config,
    mandrel_vortex_backend_t **out_backend);

void mandrel_vortex_backend_destroy(mandrel_vortex_backend_t *backend);

bool mandrel_vortex_can_offload_mul_mat(const mandrel_vortex_mul_mat_desc_t *desc);

int32_t mandrel_vortex_mul_mat_i8_i8_i32(
    mandrel_vortex_backend_t *backend,
    const mandrel_vortex_mul_mat_desc_t *desc,
    const int8_t *lhs,
    size_t lhs_len,
    const int8_t *rhs,
    size_t rhs_len,
    int32_t *out,
    size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* MANDREL_VORTEX_H */
