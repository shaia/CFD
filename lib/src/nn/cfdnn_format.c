/**
 * @file cfdnn_format.c
 * @brief Reader and writer for the portable `.cfdnn` weight format.
 *
 * Modelled directly on lib/src/io/checkpoint.c, which already solved this
 * problem for `.cfdchk`: a latching I/O helper, explicit little-endian
 * fixed-width encoding (never a raw struct fwrite), an endianness marker, a
 * table-less CRC32, and a format version that REJECTS unknown values rather
 * than guessing.
 *
 * The one deliberate deviation is that reads can come from a memory image as
 * well as a file, so cfd_nn_model_load() and cfd_nn_model_load_memory() share
 * every line of parsing code. That is what lets the test suite embed a golden
 * model as a C array instead of shipping a data file under tests/.
 *
 * There is no JSON sidecar. Everything metadata would carry lives in the same
 * byte stream under the same CRC -- a second file could disagree with the
 * first, and the checksum would not catch it. It would also require the
 * project's first third-party parser, or a hand-rolled one hardened against
 * hostile input, for no benefit.
 */

#include "cfdnn_internal.h"

#include "cfd/core/cfd_status.h"
#include "cfd/core/cfd_version.h"
#include "cfd/core/memory.h"

#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "cfdnn requires IEEE-754 binary32 float");

/* ==========================================================================
 * Format constants
 * ========================================================================== */

static const uint8_t NN_MAGIC[8] = {'C', 'F', 'D', 'N', 'N', 0, 0, 0};

#define NN_ENDIAN_MARKER 0x01020304u /* decoded LE; foreign-endian files differ */
#define NN_FLAG_CHECKSUM 0x0001u     /* flags bit0: trailing CRC32 present */
#define NN_DTYPE_F32     1u
#define NN_DTYPE_F64     2u /* reserved; rejected by this reader */

/* ==========================================================================
 * CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) -- table-less and stateless
 * so it is thread-safe with no static initialization. Lifted from
 * checkpoint.c; the ten lines are duplicated deliberately to keep io/ and nn/
 * independent of each other.
 * ========================================================================== */

static uint32_t nn_crc32_update(uint32_t crc, const void* data, size_t n) {
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
 * Latching little-endian I/O helper (file or memory)
 * ========================================================================== */

typedef struct {
    FILE*          fp;  /* NULL when reading from memory */
    const uint8_t* mem; /* memory image, or NULL */
    size_t         mem_size;
    size_t         mem_pos;
    cfd_status_t   status; /* first error latches; later ops become no-ops */
    uint32_t       crc;
} nn_io;

static void nn_write_bytes(nn_io* io, const void* p, size_t n) {
    if (io->status != CFD_SUCCESS) {
        return;
    }
    if (n && fwrite(p, 1, n, io->fp) != n) {
        io->status = CFD_ERROR_IO;
        return;
    }
    io->crc = nn_crc32_update(io->crc, p, n);
}

static void nn_read_bytes(nn_io* io, void* p, size_t n) {
    if (io->status != CFD_SUCCESS) {
        return;
    }
    if (n == 0) {
        return;
    }
    if (io->mem) {
        if (io->mem_pos + n > io->mem_size) {
            io->status = CFD_ERROR_IO; /* truncated image */
            return;
        }
        memcpy(p, io->mem + io->mem_pos, n);
        io->mem_pos += n;
    } else if (fread(p, 1, n, io->fp) != n) {
        io->status = CFD_ERROR_IO; /* truncated file */
        return;
    }
    io->crc = nn_crc32_update(io->crc, p, n);
}

/* --- fixed-width writers (explicit little-endian) --- */

static void put_u8(nn_io* io, uint8_t v) {
    nn_write_bytes(io, &v, 1);
}

static void put_u16(nn_io* io, uint16_t v) {
    uint8_t b[2] = {(uint8_t)(v & 0xFFu), (uint8_t)((v >> 8) & 0xFFu)};
    nn_write_bytes(io, b, sizeof(b));
}

static void put_u32(nn_io* io, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
    nn_write_bytes(io, b, sizeof(b));
}

static void put_f32(nn_io* io, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits)); /* IEEE-754 bit pattern, alias-safe */
    put_u32(io, bits);
}

static void put_string(nn_io* io, const char* s) {
    size_t n = s ? strlen(s) : 0;
    if (n > CFD_NN_MAX_STRING) {
        n = CFD_NN_MAX_STRING;
    }
    put_u32(io, (uint32_t)n);
    nn_write_bytes(io, s, n);
}

/* --- fixed-width readers --- */

static uint8_t get_u8(nn_io* io) {
    uint8_t v = 0;
    nn_read_bytes(io, &v, 1);
    return v;
}

static uint16_t get_u16(nn_io* io) {
    uint8_t b[2] = {0};
    nn_read_bytes(io, b, sizeof(b));
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
}

static uint32_t get_u32(nn_io* io) {
    uint8_t b[4] = {0};
    nn_read_bytes(io, b, sizeof(b));
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= ((uint32_t)b[i]) << (8 * i);
    }
    return v;
}

static float get_f32(nn_io* io) {
    uint32_t bits = get_u32(io);
    float    v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* Raw 32-bit read that does NOT fold into the running CRC -- used for the
 * trailing checksum itself, which cannot cover its own bytes. */
static uint32_t get_u32_raw(nn_io* io) {
    uint32_t saved = io->crc;
    uint32_t v     = get_u32(io);
    io->crc        = saved;
    return v;
}

static void put_u32_raw(nn_io* io, uint32_t v) {
    uint32_t saved = io->crc;
    put_u32(io, v);
    io->crc = saved;
}

/* ==========================================================================
 * Writer
 * ========================================================================== */

/** Shape validation shared by writer and reader, so an invalid model cannot be
 *  produced in the first place. */
static cfd_status_t validate_layers(const cfd_nn_layer_desc_t* layers, size_t n) {
    if (!layers || n == 0 || n > CFD_NN_MAX_LAYERS) {
        return CFD_ERROR_INVALID;
    }
    for (size_t i = 0; i < n; i++) {
        const cfd_nn_layer_desc_t* l = &layers[i];
        if (l->kind != CFD_NN_LAYER_DENSE) {
            return CFD_ERROR_INVALID;
        }
        if (l->activation < CFD_NN_ACT_IDENTITY || l->activation > CFD_NN_ACT_SOFTPLUS) {
            return CFD_ERROR_INVALID;
        }
        if (l->in_features == 0 || l->out_features == 0 ||
            l->in_features > CFD_NN_MAX_FEATURES || l->out_features > CFD_NN_MAX_FEATURES) {
            return CFD_ERROR_INVALID;
        }
        uint64_t wc = (uint64_t)l->in_features * (uint64_t)l->out_features;
        if (wc > CFD_NN_MAX_WEIGHTS || !l->weights) {
            return CFD_ERROR_INVALID;
        }
        /* Layer i's input width must equal layer i-1's output width. */
        if (i > 0 && l->in_features != layers[i - 1].out_features) {
            return CFD_ERROR_INVALID;
        }
    }
    return CFD_SUCCESS;
}

cfd_status_t cfd_nn_model_write(const char* path, const cfd_nn_model_desc_t* desc) {
    if (!path || !desc) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t vs = validate_layers(desc->layers, desc->layer_count);
    if (vs != CFD_SUCCESS) {
        return vs;
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        return CFD_ERROR_IO;
    }
    nn_io io = {fp, NULL, 0, 0, CFD_SUCCESS, 0xFFFFFFFFu};

    nn_write_bytes(&io, NN_MAGIC, sizeof(NN_MAGIC));
    put_u32(&io, CFD_NN_FORMAT_VERSION);
    put_u32(&io, NN_ENDIAN_MARKER);
    put_u16(&io, (uint16_t)CFD_VERSION_MAJOR);
    put_u16(&io, (uint16_t)CFD_VERSION_MINOR);
    put_u16(&io, (uint16_t)CFD_VERSION_PATCH);
    put_u16(&io, NN_FLAG_CHECKSUM);
    put_u8(&io, (uint8_t)NN_DTYPE_F32);
    put_u8(&io, 0);
    put_u16(&io, 0);
    put_u32(&io, (uint32_t)desc->layer_count);
    put_u32(&io, 0);
    put_u32(&io, 0);
    put_string(&io, desc->name);

    for (size_t i = 0; i < desc->layer_count; i++) {
        const cfd_nn_layer_desc_t* l  = &desc->layers[i];
        size_t                     wc = l->in_features * l->out_features;
        put_u16(&io, (uint16_t)l->kind);
        put_u16(&io, (uint16_t)l->activation);
        put_f32(&io, l->act_param);
        put_u32(&io, (uint32_t)l->in_features);
        put_u32(&io, (uint32_t)l->out_features);
        put_u32(&io, (uint32_t)wc);
        for (size_t k = 0; k < wc; k++) {
            put_f32(&io, l->weights[k]);
        }
        put_u32(&io, l->bias ? (uint32_t)l->out_features : 0u);
        if (l->bias) {
            for (size_t k = 0; k < l->out_features; k++) {
                put_f32(&io, l->bias[k]);
            }
        }
    }

    put_u32_raw(&io, io.crc ^ 0xFFFFFFFFu);

    cfd_status_t st = io.status;
    if (fclose(fp) != 0 && st == CFD_SUCCESS) {
        st = CFD_ERROR_IO;
    }
    return st;
}

/* ==========================================================================
 * Reader
 * ========================================================================== */

static void model_free(cfd_nn_model_t* m) {
    if (!m) {
        return;
    }
    cfd_free(m->name);
    cfd_free(m->blob);
    cfd_free(m->layers);
    cfd_free(m);
}

cfd_status_t cfd_nn_load_impl(const char* path, const void* bytes, size_t size,
                              cfd_nn_model_t** out_model) {
    if (!out_model) {
        return CFD_ERROR_INVALID;
    }
    *out_model = NULL;

    FILE* fp = NULL;
    if (path) {
        fp = fopen(path, "rb");
        if (!fp) {
            return CFD_ERROR_IO;
        }
    } else if (!bytes) {
        return CFD_ERROR_INVALID;
    }

    nn_io io = {fp, (const uint8_t*)bytes, size, 0, CFD_SUCCESS, 0xFFFFFFFFu};

    uint8_t magic[8] = {0};
    nn_read_bytes(&io, magic, sizeof(magic));
    if (io.status == CFD_SUCCESS && memcmp(magic, NN_MAGIC, sizeof(magic)) != 0) {
        io.status = CFD_ERROR_INVALID;
    }

    uint32_t version = get_u32(&io);
    uint32_t endian  = get_u32(&io);
    (void)get_u16(&io); /* lib major */
    (void)get_u16(&io); /* lib minor */
    (void)get_u16(&io); /* lib patch */
    uint16_t flags = get_u16(&io);
    uint8_t  dtype = get_u8(&io);
    (void)get_u8(&io);
    (void)get_u16(&io);
    uint32_t layer_count = get_u32(&io);
    (void)get_u32(&io);
    (void)get_u32(&io);

    if (io.status == CFD_SUCCESS) {
        /* Reject unknown, never guess -- the checkpoint.c rule, applied to
         * version, byte order and precision alike. */
        if (version != CFD_NN_FORMAT_VERSION || endian != NN_ENDIAN_MARKER ||
            dtype != NN_DTYPE_F32) {
            io.status = CFD_ERROR_UNSUPPORTED;
        } else if (layer_count == 0 || layer_count > CFD_NN_MAX_LAYERS) {
            io.status = CFD_ERROR_INVALID; /* cap checked before any allocation */
        }
    }
    if (io.status != CFD_SUCCESS) {
        if (fp) {
            fclose(fp);
        }
        return io.status;
    }

    cfd_nn_model_t* m = (cfd_nn_model_t*)cfd_calloc(1, sizeof(*m));
    if (!m) {
        if (fp) {
            fclose(fp);
        }
        return CFD_ERROR_NOMEM;
    }
    m->layer_count = layer_count;
    m->layers      = (cfd_nn_layer_t*)cfd_calloc(layer_count, sizeof(cfd_nn_layer_t));
    if (!m->layers) {
        model_free(m);
        if (fp) {
            fclose(fp);
        }
        return CFD_ERROR_NOMEM;
    }

    /* Model name. */
    uint32_t name_len = get_u32(&io);
    if (io.status == CFD_SUCCESS && name_len > CFD_NN_MAX_STRING) {
        io.status = CFD_ERROR_INVALID;
    }
    if (io.status == CFD_SUCCESS) {
        m->name = (char*)cfd_calloc(name_len + 1, 1);
        if (!m->name) {
            io.status = CFD_ERROR_NOMEM;
        } else {
            nn_read_bytes(&io, m->name, name_len);
        }
    }

    /* Two passes would need a seek, which the memory path would have to
     * emulate; instead layers are read straight into a growing blob whose
     * final size is known only at the end, so each layer's slice is recorded
     * as an offset and resolved to a pointer afterwards. */
    size_t*  offsets   = NULL;
    size_t*  bias_offs = NULL;
    size_t   blob_used = 0;
    size_t   blob_cap  = 0;
    float*   blob      = NULL;

    if (io.status == CFD_SUCCESS) {
        offsets   = (size_t*)cfd_calloc(layer_count, sizeof(size_t));
        bias_offs = (size_t*)cfd_calloc(layer_count, sizeof(size_t));
        if (!offsets || !bias_offs) {
            io.status = CFD_ERROR_NOMEM;
        }
    }

    for (uint32_t i = 0; i < layer_count && io.status == CFD_SUCCESS; i++) {
        uint16_t kind = get_u16(&io);
        uint16_t act  = get_u16(&io);
        float    prm  = get_f32(&io);
        uint32_t nin  = get_u32(&io);
        uint32_t nout = get_u32(&io);
        uint32_t wc   = get_u32(&io);

        if (io.status != CFD_SUCCESS) {
            break;
        }
        if (kind != (uint16_t)CFD_NN_LAYER_DENSE || act > (uint16_t)CFD_NN_ACT_SOFTPLUS) {
            io.status = CFD_ERROR_INVALID;
            break;
        }
        if (nin == 0 || nout == 0 || nin > CFD_NN_MAX_FEATURES || nout > CFD_NN_MAX_FEATURES) {
            io.status = CFD_ERROR_INVALID;
            break;
        }
        if ((uint64_t)wc != (uint64_t)nin * (uint64_t)nout || wc > CFD_NN_MAX_WEIGHTS) {
            io.status = CFD_ERROR_INVALID;
            break;
        }
        if (i > 0 && nin != (uint32_t)m->layers[i - 1].out_features) {
            io.status = CFD_ERROR_INVALID; /* shape mismatch between layers */
            break;
        }

        /* Grow the blob to hold weights (+ bias, read just below). */
        size_t need = blob_used + wc + nout;
        if (need > blob_cap) {
            size_t ncap = blob_cap ? blob_cap * 2 : 1024;
            while (ncap < need) {
                ncap *= 2;
            }
            float* nb = (float*)cfd_malloc(ncap * sizeof(float));
            if (!nb) {
                io.status = CFD_ERROR_NOMEM;
                break;
            }
            if (blob) {
                memcpy(nb, blob, blob_used * sizeof(float));
                cfd_free(blob);
            }
            blob     = nb;
            blob_cap = ncap;
        }

        offsets[i] = blob_used;
        for (uint32_t k = 0; k < wc; k++) {
            blob[blob_used + k] = get_f32(&io);
        }
        blob_used += wc;

        uint32_t bc = get_u32(&io);
        if (io.status != CFD_SUCCESS) {
            break;
        }
        if (bc != 0 && bc != nout) {
            io.status = CFD_ERROR_INVALID;
            break;
        }
        if (bc) {
            bias_offs[i] = blob_used + 1; /* +1 so 0 means "no bias" */
            for (uint32_t k = 0; k < bc; k++) {
                blob[blob_used + k] = get_f32(&io);
            }
            blob_used += bc;
        }

        m->layers[i].activation   = (cfd_nn_activation_t)act;
        m->layers[i].act_param    = prm;
        m->layers[i].in_features  = nin;
        m->layers[i].out_features = nout;
        if (nin > m->widest) {
            m->widest = nin;
        }
        if (nout > m->widest) {
            m->widest = nout;
        }
    }

    /* Trailing CRC. */
    if (io.status == CFD_SUCCESS && (flags & NN_FLAG_CHECKSUM)) {
        uint32_t expect = io.crc ^ 0xFFFFFFFFu;
        uint32_t stored = get_u32_raw(&io);
        if (io.status == CFD_SUCCESS && stored != expect) {
            io.status = CFD_ERROR_IO; /* truncation or bit-rot */
        }
    }

    if (io.status == CFD_SUCCESS) {
        for (uint32_t i = 0; i < layer_count; i++) {
            m->layers[i].weights = blob + offsets[i];
            m->layers[i].bias    = bias_offs[i] ? blob + (bias_offs[i] - 1) : NULL;
        }
        m->blob    = blob;
        m->inputs  = m->layers[0].in_features;
        m->outputs = m->layers[layer_count - 1].out_features;
    }

    cfd_free(offsets);
    cfd_free(bias_offs);
    if (fp) {
        fclose(fp);
    }

    if (io.status != CFD_SUCCESS) {
        cfd_free(blob);
        m->blob = NULL;
        model_free(m);
        return io.status;
    }

    *out_model = m;
    return CFD_SUCCESS;
}

void cfd_nn_model_destroy(cfd_nn_model_t* model) {
    model_free(model);
}
