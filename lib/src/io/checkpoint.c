/**
 * @file checkpoint.c
 * @brief Portable binary save/restore of complete simulation state.
 *
 * See checkpoint.h for the rationale. The serializer is deliberately
 * field-by-field with fixed-width little-endian encoding (no raw struct dumps),
 * so a file written on one platform reads identically on another. A small
 * latching I/O helper (chk_io) centralizes byte-order encoding, short-read /
 * short-write detection, and the running CRC32 so the high-level reader/writer
 * read like a flat description of the format.
 */

#include "cfd/io/checkpoint.h"

#include "cfd/core/cfd_version.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ==========================================================================
 * Format constants
 * ========================================================================== */

static const uint8_t CHK_MAGIC[8] = {'C', 'F', 'D', 'C', 'H', 'K', 0, 0};

#define CHK_ENDIAN_MARKER 0x01020304u  /* decoded LE; foreign-endian files differ */
#define CHK_FLAG_CHECKSUM 0x0001u      /* header flags bit0: trailing CRC32 present */

/* Robustness caps for malformed/hostile files (not physical limits). */
#define CHK_MAX_DIM    (1u << 24)  /* per-axis point count */
#define CHK_MAX_STRING (1u << 20)  /* length-prefixed string bytes */

/* ==========================================================================
 * CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) — table-less and stateless
 * so it is thread-safe with no static initialization.
 * ========================================================================== */

static uint32_t crc32_update(uint32_t crc, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

/* ==========================================================================
 * Latching little-endian I/O helper
 * ========================================================================== */

typedef struct {
    FILE* fp;
    cfd_status_t status; /* first error latches; later ops become no-ops */
    uint32_t crc;        /* running raw CRC (final value is crc ^ 0xFFFFFFFF) */
} chk_io;

static void chk_write_bytes(chk_io* io, const void* p, size_t n) {
    if (io->status != CFD_SUCCESS) {
        return;
    }
    if (n && fwrite(p, 1, n, io->fp) != n) {
        io->status = CFD_ERROR_IO;
        return;
    }
    io->crc = crc32_update(io->crc, p, n);
}

static void chk_read_bytes(chk_io* io, void* p, size_t n) {
    if (io->status != CFD_SUCCESS) {
        return;
    }
    if (n && fread(p, 1, n, io->fp) != n) {
        io->status = CFD_ERROR_IO;
        return;
    }
    io->crc = crc32_update(io->crc, p, n);
}

/* --- fixed-width writers (explicit little-endian) --- */

static void put_u16(chk_io* io, uint16_t v) {
    uint8_t b[2] = {(uint8_t)(v & 0xFFu), (uint8_t)((v >> 8) & 0xFFu)};
    chk_write_bytes(io, b, sizeof(b));
}

static void put_u32(chk_io* io, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
    chk_write_bytes(io, b, sizeof(b));
}

static void put_u64(chk_io* io, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) {
        b[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
    chk_write_bytes(io, b, sizeof(b));
}

static void put_i32(chk_io* io, int32_t v) {
    put_u32(io, (uint32_t)v);
}

static void put_f64(chk_io* io, double v) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits)); /* IEEE-754 bit pattern, alias-safe */
    put_u64(io, bits);
}

static void put_f64_array(chk_io* io, const double* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        put_f64(io, a[i]);
    }
}

/* NULL or empty string both serialize as length 0. */
static void put_string(chk_io* io, const char* s) {
    uint32_t len = s ? (uint32_t)strlen(s) : 0u;
    put_u32(io, len);
    if (len) {
        chk_write_bytes(io, s, len);
    }
}

/* --- fixed-width readers (explicit little-endian) --- */

static uint16_t get_u16(chk_io* io) {
    uint8_t b[2] = {0, 0};
    chk_read_bytes(io, b, sizeof(b));
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
}

static uint32_t get_u32(chk_io* io) {
    uint8_t b[4] = {0};
    chk_read_bytes(io, b, sizeof(b));
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= (uint32_t)b[i] << (8 * i);
    }
    return v;
}

static uint64_t get_u64(chk_io* io) {
    uint8_t b[8] = {0};
    chk_read_bytes(io, b, sizeof(b));
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t)b[i] << (8 * i);
    }
    return v;
}

static int32_t get_i32(chk_io* io) {
    return (int32_t)get_u32(io);
}

static double get_f64(chk_io* io) {
    uint64_t bits = get_u64(io);
    double v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static void get_f64_array(chk_io* io, double* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = get_f64(io);
    }
}

/* Reads a length-prefixed string. If buf is NULL the bytes are consumed (and
 * folded into the CRC) but discarded. Sets CFD_ERROR_INVALID if the stored
 * length is implausible or does not fit in the caller's buffer. */
static void get_string(chk_io* io, char* buf, size_t cap) {
    uint32_t len = get_u32(io);
    if (io->status != CFD_SUCCESS) {
        return;
    }
    if (len > CHK_MAX_STRING) {
        io->status = CFD_ERROR_INVALID;
        return;
    }
    // A caller buffer with zero capacity cannot be filled or NUL-terminated, so
    // treat it as invalid rather than silently discarding the string bytes.
    if (buf && (cap == 0 || (size_t)len + 1 > cap)) {
        io->status = CFD_ERROR_INVALID;
        return;
    }

    uint8_t tmp[256];
    uint32_t remaining = len;
    size_t off = 0;
    while (remaining > 0 && io->status == CFD_SUCCESS) {
        size_t chunk = remaining < sizeof(tmp) ? remaining : sizeof(tmp);
        chk_read_bytes(io, tmp, chunk);
        if (io->status != CFD_SUCCESS) {
            break;
        }
        if (buf && cap > 0) {
            memcpy(buf + off, tmp, chunk);
            off += chunk;
        }
        remaining -= (uint32_t)chunk;
    }
    if (buf && cap > 0 && io->status == CFD_SUCCESS) {
        buf[len] = '\0';
    }
}

/* Trailing CRC32 is written/read raw (NOT folded into the running CRC). */
static void put_u32_raw(chk_io* io, uint32_t v) {
    if (io->status != CFD_SUCCESS) {
        return;
    }
    uint8_t b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
    if (fwrite(b, 1, sizeof(b), io->fp) != sizeof(b)) {
        io->status = CFD_ERROR_IO;
    }
}

static uint32_t get_u32_raw(chk_io* io) {
    uint8_t b[4] = {0};
    if (io->status != CFD_SUCCESS) {
        return 0;
    }
    if (fread(b, 1, sizeof(b), io->fp) != sizeof(b)) {
        io->status = CFD_ERROR_IO;
        return 0;
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= (uint32_t)b[i] << (8 * i);
    }
    return v;
}

/* ==========================================================================
 * Section writers (shared layout description used by both write and read)
 * ========================================================================== */

static void write_header(chk_io* io) {
    chk_write_bytes(io, CHK_MAGIC, sizeof(CHK_MAGIC));
    put_u32(io, CFD_CHECKPOINT_FORMAT_VERSION);
    put_u32(io, CHK_ENDIAN_MARKER);
    put_u16(io, (uint16_t)CFD_VERSION_MAJOR);
    put_u16(io, (uint16_t)CFD_VERSION_MINOR);
    put_u16(io, (uint16_t)CFD_VERSION_PATCH);
    put_u16(io, (uint16_t)CHK_FLAG_CHECKSUM);
    put_u64(io, 0); /* reserved */
}

static void write_grid(chk_io* io, const grid* g) {
    put_u64(io, (uint64_t)g->nx);
    put_u64(io, (uint64_t)g->ny);
    put_u64(io, (uint64_t)g->nz);
    put_f64(io, g->xmin);
    put_f64(io, g->xmax);
    put_f64(io, g->ymin);
    put_f64(io, g->ymax);
    put_f64(io, g->zmin);
    put_f64(io, g->zmax);
    put_f64_array(io, g->x, g->nx);
    put_f64_array(io, g->y, g->ny);
    put_f64_array(io, g->dx, g->nx - 1);
    put_f64_array(io, g->dy, g->ny - 1);
    if (g->nz > 1) {
        put_f64_array(io, g->z, g->nz);
        put_f64_array(io, g->dz, g->nz - 1);
        put_f64(io, g->inv_dz2); /* derived but min-dz dependent; store to stay exact */
    }
}

static void write_field(chk_io* io, const flow_field* f) {
    size_t n = f->nx * f->ny * f->nz;
    put_u64(io, (uint64_t)f->nx); /* self-check against grid on read */
    put_u64(io, (uint64_t)f->ny);
    put_u64(io, (uint64_t)f->nz);
    put_f64_array(io, f->u, n);
    put_f64_array(io, f->v, n);
    put_f64_array(io, f->w, n);
    put_f64_array(io, f->p, n);
    put_f64_array(io, f->rho, n);
    put_f64_array(io, f->T, n);
}

static void write_params(chk_io* io, const ns_solver_params_t* p) {
    put_f64(io, p->dt);
    put_f64(io, p->cfl);
    put_f64(io, p->gamma);
    put_f64(io, p->mu);
    put_f64(io, p->k);
    put_i32(io, p->max_iter);
    put_f64(io, p->tolerance);
    put_f64(io, p->source_amplitude_u);
    put_f64(io, p->source_amplitude_v);
    put_f64(io, p->source_decay_rate);
    put_f64(io, p->pressure_coupling);
    put_f64(io, p->alpha);
    put_f64(io, p->beta);
    put_f64(io, p->T_ref);
    put_f64(io, p->gravity[0]);
    put_f64(io, p->gravity[1]);
    put_f64(io, p->gravity[2]);
    /* thermal_bc: face types then Dirichlet values */
    put_i32(io, (int32_t)p->thermal_bc.left);
    put_i32(io, (int32_t)p->thermal_bc.right);
    put_i32(io, (int32_t)p->thermal_bc.bottom);
    put_i32(io, (int32_t)p->thermal_bc.top);
    put_i32(io, (int32_t)p->thermal_bc.front);
    put_i32(io, (int32_t)p->thermal_bc.back);
    put_f64(io, p->thermal_bc.dirichlet_values.left);
    put_f64(io, p->thermal_bc.dirichlet_values.right);
    put_f64(io, p->thermal_bc.dirichlet_values.top);
    put_f64(io, p->thermal_bc.dirichlet_values.bottom);
    put_f64(io, p->thermal_bc.dirichlet_values.front);
    put_f64(io, p->thermal_bc.dirichlet_values.back);
}

/* ==========================================================================
 * Public API: write
 * ========================================================================== */

cfd_status_t cfd_checkpoint_write(const char* path,
                                  const grid* g,
                                  const flow_field* field,
                                  const ns_solver_params_t* params,
                                  double current_time,
                                  const char* solver_name,
                                  const char* run_prefix,
                                  const char* output_base_dir) {
    if (!path || !g || !field || !params || !solver_name) {
        cfd_set_error(CFD_ERROR_INVALID, "cfd_checkpoint_write: NULL argument");
        return CFD_ERROR_INVALID;
    }
    if (field->nx != g->nx || field->ny != g->ny || field->nz != g->nz) {
        cfd_set_error(CFD_ERROR_INVALID, "cfd_checkpoint_write: field/grid dimension mismatch");
        return CFD_ERROR_INVALID;
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        cfd_set_error(CFD_ERROR_IO, "cfd_checkpoint_write: failed to open file");
        return CFD_ERROR_IO;
    }

    chk_io io = {fp, CFD_SUCCESS, 0xFFFFFFFFu};

    write_header(&io);
    write_grid(&io, g);
    write_field(&io, field);
    write_params(&io, params);
    put_f64(&io, current_time);
    put_string(&io, solver_name);
    put_string(&io, run_prefix);
    put_string(&io, output_base_dir);
    put_u32_raw(&io, io.crc ^ 0xFFFFFFFFu); /* trailing CRC32 over the body */

    cfd_status_t status = io.status;
    if (fclose(fp) != 0 && status == CFD_SUCCESS) {
        status = CFD_ERROR_IO;
    }
    if (status != CFD_SUCCESS) {
        cfd_set_error(status, "cfd_checkpoint_write: write failed");
    }
    return status;
}

/* ==========================================================================
 * Public API: read
 * ========================================================================== */

cfd_status_t cfd_checkpoint_read(const char* path,
                                 grid** out_grid,
                                 flow_field** out_field,
                                 ns_solver_params_t* out_params,
                                 double* out_current_time,
                                 char* out_solver_name,
                                 size_t solver_name_cap,
                                 char* out_run_prefix,
                                 size_t run_prefix_cap,
                                 char* out_output_base_dir,
                                 size_t output_base_dir_cap) {
    if (out_grid) {
        *out_grid = NULL;
    }
    if (out_field) {
        *out_field = NULL;
    }
    if (!path || !out_grid || !out_field || !out_params) {
        cfd_set_error(CFD_ERROR_INVALID, "cfd_checkpoint_read: NULL argument");
        return CFD_ERROR_INVALID;
    }
    memset(out_params, 0, sizeof(*out_params)); /* leaves callback pointers NULL */

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        cfd_set_error(CFD_ERROR_IO, "cfd_checkpoint_read: failed to open file");
        return CFD_ERROR_IO;
    }

    chk_io io = {fp, CFD_SUCCESS, 0xFFFFFFFFu};
    grid* g = NULL;
    flow_field* f = NULL;

    /* --- header --- */
    uint8_t magic[8] = {0};
    chk_read_bytes(&io, magic, sizeof(magic));
    if (io.status == CFD_SUCCESS && memcmp(magic, CHK_MAGIC, sizeof(magic)) != 0) {
        io.status = CFD_ERROR_INVALID;
    }
    uint32_t version = get_u32(&io);
    uint32_t endian = get_u32(&io);
    (void)get_u16(&io); /* lib major */
    (void)get_u16(&io); /* lib minor */
    (void)get_u16(&io); /* lib patch */
    uint16_t flags = get_u16(&io);
    (void)get_u64(&io); /* reserved */
    if (io.status == CFD_SUCCESS &&
        (version != CFD_CHECKPOINT_FORMAT_VERSION || endian != CHK_ENDIAN_MARKER)) {
        io.status = CFD_ERROR_UNSUPPORTED;
    }

    /* --- grid --- */
    uint64_t nx = get_u64(&io);
    uint64_t ny = get_u64(&io);
    uint64_t nz = get_u64(&io);
    double xmin = get_f64(&io);
    double xmax = get_f64(&io);
    double ymin = get_f64(&io);
    double ymax = get_f64(&io);
    double zmin = get_f64(&io);
    double zmax = get_f64(&io);
    if (io.status == CFD_SUCCESS &&
        (nx < 2 || ny < 2 || nz < 1 || nx > CHK_MAX_DIM || ny > CHK_MAX_DIM ||
         nz > CHK_MAX_DIM)) {
        io.status = CFD_ERROR_INVALID;
    }
    if (io.status == CFD_SUCCESS) {
        g = grid_create((size_t)nx, (size_t)ny, (size_t)nz, xmin, xmax, ymin, ymax, zmin, zmax);
        if (!g) {
            io.status = CFD_ERROR_NOMEM;
        }
    }
    if (io.status == CFD_SUCCESS) {
        get_f64_array(&io, g->x, g->nx);
        get_f64_array(&io, g->y, g->ny);
        get_f64_array(&io, g->dx, g->nx - 1);
        get_f64_array(&io, g->dy, g->ny - 1);
        if (g->nz > 1) {
            get_f64_array(&io, g->z, g->nz);
            get_f64_array(&io, g->dz, g->nz - 1);
            g->inv_dz2 = get_f64(&io);
        }
    }

    /* --- field (dimensions must match grid) --- */
    uint64_t fnx = get_u64(&io);
    uint64_t fny = get_u64(&io);
    uint64_t fnz = get_u64(&io);
    if (io.status == CFD_SUCCESS && (fnx != nx || fny != ny || fnz != nz)) {
        io.status = CFD_ERROR_INVALID;
    }
    if (io.status == CFD_SUCCESS) {
        f = flow_field_create((size_t)nx, (size_t)ny, (size_t)nz);
        if (!f) {
            io.status = CFD_ERROR_NOMEM;
        }
    }
    if (io.status == CFD_SUCCESS) {
        size_t n = f->nx * f->ny * f->nz;
        get_f64_array(&io, f->u, n);
        get_f64_array(&io, f->v, n);
        get_f64_array(&io, f->w, n);
        get_f64_array(&io, f->p, n);
        get_f64_array(&io, f->rho, n);
        get_f64_array(&io, f->T, n);
    }

    /* --- params (scalars only; callbacks remain NULL from the memset) --- */
    out_params->dt = get_f64(&io);
    out_params->cfl = get_f64(&io);
    out_params->gamma = get_f64(&io);
    out_params->mu = get_f64(&io);
    out_params->k = get_f64(&io);
    out_params->max_iter = get_i32(&io);
    out_params->tolerance = get_f64(&io);
    out_params->source_amplitude_u = get_f64(&io);
    out_params->source_amplitude_v = get_f64(&io);
    out_params->source_decay_rate = get_f64(&io);
    out_params->pressure_coupling = get_f64(&io);
    out_params->alpha = get_f64(&io);
    out_params->beta = get_f64(&io);
    out_params->T_ref = get_f64(&io);
    out_params->gravity[0] = get_f64(&io);
    out_params->gravity[1] = get_f64(&io);
    out_params->gravity[2] = get_f64(&io);
    out_params->thermal_bc.left = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.right = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.bottom = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.top = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.front = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.back = (bc_type_t)get_i32(&io);
    out_params->thermal_bc.dirichlet_values.left = get_f64(&io);
    out_params->thermal_bc.dirichlet_values.right = get_f64(&io);
    out_params->thermal_bc.dirichlet_values.top = get_f64(&io);
    out_params->thermal_bc.dirichlet_values.bottom = get_f64(&io);
    out_params->thermal_bc.dirichlet_values.front = get_f64(&io);
    out_params->thermal_bc.dirichlet_values.back = get_f64(&io);

    /* --- sim metadata --- */
    double current_time = get_f64(&io);
    get_string(&io, out_solver_name, solver_name_cap);
    get_string(&io, out_run_prefix, run_prefix_cap);
    get_string(&io, out_output_base_dir, output_base_dir_cap);

    /* --- trailing CRC32 over the body --- */
    if (flags & CHK_FLAG_CHECKSUM) {
        uint32_t computed = io.crc ^ 0xFFFFFFFFu;
        uint32_t stored = get_u32_raw(&io);
        if (io.status == CFD_SUCCESS && computed != stored) {
            io.status = CFD_ERROR_IO;
        }
    }

    fclose(fp);

    if (io.status != CFD_SUCCESS) {
        grid_destroy(g);
        flow_field_destroy(f);
        cfd_set_error(io.status, "cfd_checkpoint_read: read failed");
        return io.status;
    }

    *out_grid = g;
    *out_field = f;
    if (out_current_time) {
        *out_current_time = current_time;
    }
    return CFD_SUCCESS;
}
