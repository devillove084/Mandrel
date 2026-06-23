#ifndef MANDREL_VORTEX_H
#define MANDREL_VORTEX_H

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

typedef struct mandrel_vortex_backend_config {
    size_t struct_size;
    const char *runtime_library_path; /* optional; NULL uses Rust-side runtime search */
    uint32_t device_index;            /* usually 0 */
    uint32_t flags;                   /* reserved, must be 0 */
} mandrel_vortex_backend_config_t;

const char *mandrel_vortex_backend_name(void);
const char *mandrel_vortex_status_message(int32_t status);

/* Thread-local; valid until the next mandrel_vortex_* call on the same thread. */
const char *mandrel_vortex_last_error_message(void);

int32_t mandrel_vortex_backend_create(
    const mandrel_vortex_backend_config_t *config,
    mandrel_vortex_backend_t **out_backend);

void mandrel_vortex_backend_destroy(mandrel_vortex_backend_t *backend);

#ifdef __cplusplus
}
#endif

#endif /* MANDREL_VORTEX_H */
