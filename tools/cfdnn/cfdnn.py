"""Write, read and evaluate `.cfdnn` models from Python.

This is developer tooling, outside the build and outside CI. The C library is
the normative implementation of the format (lib/src/nn/cfdnn_format.c, contract
in lib/include/cfd/nn/cfdnn.h); this module mirrors it so a model trained in
Python can be handed to the solver. What keeps the two honest:

- The reader here re-implements every check the C reader makes, so a file this
  module writes and reads back has already passed the same validation.
- tests/nn/test_cfdnn_python_export.c embeds a model written by this module and
  checks the C reader loads it and the C kernels reproduce `predict()` below.
  That is the only CI coverage the exporter has; see the design note's
  Limitations section for why a Python step is not in CI.

Only numpy is required. PyTorch is optional and never imported here:
`from_torch_sequential()` duck-types the modules it is handed.

The format, in brief (little-endian throughout; CRC32 is zlib's):

     0  8  magic "CFDNN\\0\\0\\0"
     8  4  format_version u32 = 1
    12  4  endian_marker  u32 = 0x01020304
    16  6  lib_version    u16 x3
    22  2  flags          u16  bit0 = trailing CRC32
    24  1  dtype          u8   1 = f32
    25  1  tensor_layout  u8   0 = row-major
    26  2  reserved       u16 = 0
    28  4  layer_count    u32
    32  8  reserved       u32 x2 = 0
    40 ..  name (u32 len + bytes), then per Dense layer:
           kind u16 | activation u16 | act_param f32 | in u32 | out u32
           | weight_count u32 | f32[out][in] | bias_count u32 | f32[bias_count]
    .. 4  crc32 over everything before it
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np

# --- constants mirrored from cfdnn.h / cfdnn_internal.h / cfdnn_format.c ----

MAGIC = b"CFDNN\x00\x00\x00"
FORMAT_VERSION = 1
ENDIAN_MARKER = 0x01020304
FLAG_CHECKSUM = 0x0001
DTYPE_F32 = 1
LAYOUT_ROW_MAJOR = 0

LAYER_DENSE = 1

ACT_IDENTITY = 0
ACT_RELU = 1
ACT_LEAKY_RELU = 2
ACT_TANH = 3
ACT_SIGMOID = 4
ACT_SOFTPLUS = 5

ACTIVATIONS = {
    "identity": ACT_IDENTITY,
    "relu": ACT_RELU,
    "leaky_relu": ACT_LEAKY_RELU,
    "tanh": ACT_TANH,
    "sigmoid": ACT_SIGMOID,
    "softplus": ACT_SOFTPLUS,
}

MAX_LAYERS = 64
MAX_FEATURES = 4096
MAX_WEIGHTS = 1 << 22
MAX_STRING = 1 << 12

# The writer's library version is informational: the C reader skips it. It
# records which release of the format contract the exporter was written
# against, the same thing the C writer records from cfd_version.h.
LIB_VERSION = (0, 3, 0)

_HEADER = struct.Struct("<8sII3HHBBHIII")  # 40 bytes, offsets 0..39
_LAYER = struct.Struct("<HHfIII")          # kind..weight_count, 20 bytes


class CfdnnError(ValueError):
    """A model the C reader would refuse. `status` names the cfd_status_t."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass
class Dense:
    """One Dense layer. `weight` is [out][in], exactly PyTorch's Linear.weight."""

    weight: np.ndarray
    bias: Optional[np.ndarray] = None
    activation: Union[int, str] = ACT_IDENTITY  # a code, or a key of ACTIVATIONS
    act_param: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.activation, str):
            self.activation = ACTIVATIONS[self.activation]
        self.weight = np.ascontiguousarray(self.weight, dtype="<f4")
        if self.bias is not None:
            self.bias = np.ascontiguousarray(self.bias, dtype="<f4").reshape(-1)

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])


# ------------------------------------------------------------------ validate --


def validate(layers: Sequence[Dense], name: str = "") -> None:
    """The C writer's validate_layers(), plus the name cap. Raises CfdnnError."""
    if len(name.encode("utf-8")) > MAX_STRING:
        raise CfdnnError("CFD_ERROR_INVALID", "name longer than MAX_STRING")
    if not layers or len(layers) > MAX_LAYERS:
        raise CfdnnError("CFD_ERROR_INVALID", f"layer count {len(layers)} not in 1..{MAX_LAYERS}")
    for i, l in enumerate(layers):
        if l.weight.ndim != 2:
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: weight must be 2-D [out][in]")
        if l.activation not in ACTIVATIONS.values():
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: unknown activation {l.activation}")
        nin, nout = l.in_features, l.out_features
        if not (0 < nin <= MAX_FEATURES and 0 < nout <= MAX_FEATURES):
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: {nin}->{nout} beyond MAX_FEATURES")
        if nin * nout > MAX_WEIGHTS:
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: weight count beyond MAX_WEIGHTS")
        if l.bias is not None and l.bias.shape != (nout,):
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: bias must have {nout} values")
        if i > 0 and nin != layers[i - 1].out_features:
            raise CfdnnError("CFD_ERROR_INVALID",
                             f"layer {i}: input width {nin} != previous output width "
                             f"{layers[i - 1].out_features}")
        if not (np.all(np.isfinite(l.weight)) and (l.bias is None or np.all(np.isfinite(l.bias)))):
            # The C writer does not check this, but a non-finite weight can only
            # ever make predict fail with CFD_ERROR_DIVERGED; refusing it here
            # moves the failure to export time, where the cause is obvious.
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: non-finite weight or bias")


# ------------------------------------------------------------------- write --


def to_bytes(layers: Sequence[Dense], name: str = "") -> bytes:
    """Serialize a model. Byte-identical to cfd_nn_model_write() for the same input."""
    validate(layers, name)
    name_b = name.encode("utf-8")
    out = bytearray()
    out += _HEADER.pack(MAGIC, FORMAT_VERSION, ENDIAN_MARKER, *LIB_VERSION,
                        FLAG_CHECKSUM, DTYPE_F32, LAYOUT_ROW_MAJOR, 0,
                        len(layers), 0, 0)
    out += struct.pack("<I", len(name_b)) + name_b
    for l in layers:
        out += _LAYER.pack(LAYER_DENSE, l.activation, l.act_param,
                           l.in_features, l.out_features, l.weight.size)
        out += l.weight.tobytes(order="C")  # already '<f4', row-major: no transpose
        if l.bias is None:
            out += struct.pack("<I", 0)
        else:
            out += struct.pack("<I", l.out_features) + l.bias.tobytes()
    # zlib.crc32 is the C file's reflected 0xEDB88320 CRC with the same
    # 0xFFFFFFFF init and final xor, so it is the stored value directly.
    out += struct.pack("<I", zlib.crc32(out) & 0xFFFFFFFF)
    return bytes(out)


def write(path: str, layers: Sequence[Dense], name: str = "") -> None:
    data = to_bytes(layers, name)
    with open(path, "wb") as fp:
        fp.write(data)


# -------------------------------------------------------------------- read --


def from_bytes(data: bytes) -> tuple[list[Dense], str]:
    """Parse and validate a model, refusing whatever cfd_nn_load_impl() refuses.

    Statuses match the C reader's, so a test here predicts what the library
    will say about the same bytes.
    """
    if len(data) < _HEADER.size:
        raise CfdnnError("CFD_ERROR_IO", "truncated header")
    (magic, version, endian, _maj, _min, _pat, flags, dtype, layout, rsv16,
     layer_count, rsv_a, rsv_b) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise CfdnnError("CFD_ERROR_INVALID", "bad magic")
    if (version != FORMAT_VERSION or endian != ENDIAN_MARKER or dtype != DTYPE_F32
            or flags & ~FLAG_CHECKSUM & 0xFFFF or layout != LAYOUT_ROW_MAJOR
            or rsv16 or rsv_a or rsv_b):
        raise CfdnnError("CFD_ERROR_UNSUPPORTED",
                         "version, byte order, dtype, flag bits, layout or reserved word")
    if layer_count == 0 or layer_count > MAX_LAYERS:
        raise CfdnnError("CFD_ERROR_INVALID", f"layer count {layer_count}")

    pos = _HEADER.size

    def take(n: int) -> bytes:
        nonlocal pos
        if pos + n > len(data):
            raise CfdnnError("CFD_ERROR_IO", "truncated")
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    (name_len,) = struct.unpack("<I", take(4))
    if name_len > MAX_STRING:
        raise CfdnnError("CFD_ERROR_INVALID", "name too long")
    name = take(name_len).decode("utf-8", errors="replace")

    layers: list[Dense] = []
    for i in range(layer_count):
        kind, act, prm, nin, nout, wc = _LAYER.unpack(take(_LAYER.size))
        if kind != LAYER_DENSE or act > ACT_SOFTPLUS:
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: kind {kind} / activation {act}")
        if not (0 < nin <= MAX_FEATURES and 0 < nout <= MAX_FEATURES):
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: size")
        if wc != nin * nout or wc > MAX_WEIGHTS:
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: weight count")
        if i > 0 and nin != layers[-1].out_features:
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: shape mismatch")
        w = np.frombuffer(take(4 * wc), dtype="<f4").reshape(nout, nin)
        (bc,) = struct.unpack("<I", take(4))
        if bc not in (0, nout):
            raise CfdnnError("CFD_ERROR_INVALID", f"layer {i}: bias count")
        b = np.frombuffer(take(4 * bc), dtype="<f4") if bc else None
        layers.append(Dense(w, b, act, prm))

    if flags & FLAG_CHECKSUM:
        body_end = pos
        (stored,) = struct.unpack("<I", take(4))
        if stored != zlib.crc32(data[:body_end]) & 0xFFFFFFFF:
            raise CfdnnError("CFD_ERROR_IO", "CRC mismatch")
    return layers, name


def read(path: str) -> tuple[list[Dense], str]:
    with open(path, "rb") as fp:
        return from_bytes(fp.read())


# ------------------------------------------------------------------ predict --


def _activate(act: int, param: float, v: np.ndarray) -> np.ndarray:
    if act == ACT_IDENTITY:
        return v
    if act == ACT_RELU:
        return np.maximum(v, 0.0)
    if act == ACT_LEAKY_RELU:
        return np.where(v < 0.0, v * param, v)
    if act == ACT_TANH:
        return np.tanh(v)
    if act == ACT_SIGMOID:
        return 1.0 / (1.0 + np.exp(-v))
    if act == ACT_SOFTPLUS:
        return np.maximum(v, 0.0) + np.log1p(np.exp(-np.abs(v)))  # the C kernel's form
    raise CfdnnError("CFD_ERROR_INVALID", f"unknown activation {act}")


def predict(layers: Sequence[Dense], x: np.ndarray) -> np.ndarray:
    """Reference forward pass, [batch][in] -> [batch][out].

    Evaluated in float64 on the stored float32 weights, so it is the exact value
    the C kernels (which accumulate in float32) approximate -- the right
    yardstick for a tolerance, rather than a second float32 implementation that
    could share their rounding.
    """
    h = np.atleast_2d(np.asarray(x, dtype=np.float64))
    for l in layers:
        h = h @ l.weight.astype(np.float64).T
        if l.bias is not None:
            h = h + l.bias.astype(np.float64)
        h = _activate(l.activation, float(l.act_param), h)
    return h


# ------------------------------------------------------------------ folding --


def fold_input_normalization(layers: Sequence[Dense], mean: Iterable[float],
                             std: Iterable[float]) -> list[Dense]:
    """Fold x_n = (x - mean) / std into the first layer, which is exact.

    The format has no normalization layer and should not grow one: the closure
    hands the network raw features (ln S*, ln Re_t, ln nu_t/nu), so a model
    trained on standardized inputs must absorb the standardization here.
      W' = W / std   (column-wise)      b' = b - W' @ mean
    """
    mean = np.asarray(list(mean), dtype=np.float64)
    std = np.asarray(list(std), dtype=np.float64)
    first = layers[0]
    if mean.shape != (first.in_features,) or std.shape != (first.in_features,):
        raise CfdnnError("CFD_ERROR_INVALID", "mean/std must have one entry per input feature")
    if np.any(std <= 0.0) or not np.all(np.isfinite(std)):
        raise CfdnnError("CFD_ERROR_INVALID", "std must be finite and positive")
    w = first.weight.astype(np.float64) / std
    b = (first.bias.astype(np.float64) if first.bias is not None else 0.0) - w @ mean
    return [Dense(w, b, first.activation, first.act_param), *layers[1:]]


def fold_batchnorm(dense: Dense, gamma: np.ndarray, beta: np.ndarray,
                   running_mean: np.ndarray, running_var: np.ndarray,
                   eps: float = 1e-5) -> Dense:
    """Fold an inference-mode BatchNorm1d that FOLLOWS `dense` into it. Exact.

    y = gamma * (Wx + b - mu) / sqrt(var + eps) + beta
      = (s * W) x + (s * (b - mu) + beta),   s = gamma / sqrt(var + eps)

    The format rejects a batch-norm layer kind by design, so this is the only
    way a batch-norm model reaches the solver. The activation of `dense` must
    come AFTER the batch-norm for this to be valid, which is why the torch
    adapter only folds a BatchNorm1d that sits directly after a Linear.
    """
    s = np.asarray(gamma, np.float64) / np.sqrt(np.asarray(running_var, np.float64) + eps)
    b0 = dense.bias.astype(np.float64) if dense.bias is not None else 0.0
    w = dense.weight.astype(np.float64) * s[:, None]
    b = s * (b0 - np.asarray(running_mean, np.float64)) + np.asarray(beta, np.float64)
    return Dense(w, b, dense.activation, dense.act_param)


# -------------------------------------------------------------------- torch --


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def from_torch_sequential(model) -> list[Dense]:
    """Convert a torch.nn.Sequential of Linear / activation / BatchNorm1d / Dropout.

    Duck-typed on class names so this module never imports torch. Anything
    else is refused rather than skipped: an unrecognised module is a layer the
    C side would not run, and dropping it silently is the "loads, runs, and
    predicts garbage" failure the format was designed against.

    Call model.eval() first; it is asserted, because a training-mode
    BatchNorm1d uses batch statistics the file cannot record.
    """
    if getattr(model, "training", False):
        raise CfdnnError("CFD_ERROR_INVALID", "call model.eval() before export")
    acts = {"ReLU": ACT_RELU, "Tanh": ACT_TANH, "Sigmoid": ACT_SIGMOID,
            "Softplus": ACT_SOFTPLUS, "LeakyReLU": ACT_LEAKY_RELU}
    layers: list[Dense] = []
    pending_act_ok = False  # True while the last Linear has no activation yet
    for m in model:
        kind = type(m).__name__
        if kind == "Linear":
            b = _np(m.bias) if getattr(m, "bias", None) is not None else None
            layers.append(Dense(_np(m.weight), b))
            pending_act_ok = True
        elif kind in ("Dropout", "Identity"):
            continue  # identity at inference
        elif kind == "BatchNorm1d":
            if not layers or not pending_act_ok:
                raise CfdnnError("CFD_ERROR_INVALID",
                                 "BatchNorm1d must directly follow a Linear to be folded")
            if not getattr(m, "track_running_stats", True) or m.running_mean is None:
                raise CfdnnError("CFD_ERROR_INVALID", "BatchNorm1d without running statistics")
            gamma = _np(m.weight) if getattr(m, "affine", True) else np.ones(m.num_features)
            beta = _np(m.bias) if getattr(m, "affine", True) else np.zeros(m.num_features)
            layers[-1] = fold_batchnorm(layers[-1], gamma, beta, _np(m.running_mean),
                                        _np(m.running_var), float(m.eps))
        elif kind in acts:
            if not layers or not pending_act_ok:
                raise CfdnnError("CFD_ERROR_INVALID",
                                 f"{kind} must follow a Linear (activations are fused)")
            # The C kernel is exact log1p(exp(x)). torch's `threshold` only
            # switches to identity where the two agree to ~exp(-threshold), but
            # a non-unit beta is a different function.
            if kind == "Softplus" and getattr(m, "beta", 1) != 1:
                raise CfdnnError("CFD_ERROR_INVALID", "Softplus(beta != 1) is not supported")
            layers[-1].activation = acts[kind]
            if kind == "LeakyReLU":
                layers[-1].act_param = float(m.negative_slope)
            pending_act_ok = False
        else:
            raise CfdnnError("CFD_ERROR_INVALID", f"unsupported module {kind}")
    validate(layers)
    return layers


# ------------------------------------------------------------------ C header --


def to_c_header(data: bytes, symbol: str, generator: str) -> str:
    """Render model bytes as a C array, the way the test suite embeds models."""
    lines = [
        f"/* Generated by {generator}. Do not edit: regenerate instead. */",
        "",
        f"static const unsigned char {symbol}[{len(data)}] = {{",
    ]
    for i in range(0, len(data), 12):
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ",")
    lines.append("};")
    return "\n".join(lines) + "\n"
