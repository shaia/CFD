"""Tests for the Python side of the `.cfdnn` format. Run: python -m unittest tools/cfdnn/test_cfdnn.py

The C side of the contract -- that the library loads what this module writes and
its kernels reproduce predict() -- is tests/nn/test_cfdnn_python_export.c.
"""

from __future__ import annotations

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfdnn  # noqa: E402
from cfdnn import CfdnnError, Dense  # noqa: E402


def crc32_like_c(data: bytes) -> int:
    """Line-for-line port of nn_crc32_update() plus the writer's init/final xor."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1) & 0xFFFFFFFF))
    return crc ^ 0xFFFFFFFF


def small_model() -> list[Dense]:
    rng = np.random.default_rng(1)
    return [
        Dense(rng.normal(size=(4, 3)), rng.normal(size=4), "tanh"),
        Dense(rng.normal(size=(2, 4)), None, "leaky_relu", 0.1),
        Dense(rng.normal(size=(1, 2)), rng.normal(size=1), "softplus"),
    ]


def status_of(fn) -> str:
    try:
        fn()
    except CfdnnError as e:
        return e.status
    return "CFD_SUCCESS"


class FormatTests(unittest.TestCase):
    def test_header_layout_matches_the_documented_offsets(self):
        data = cfdnn.to_bytes(small_model(), "abc")
        self.assertEqual(data[0:8], b"CFDNN\x00\x00\x00")
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 1)
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], 0x01020304)
        self.assertEqual(struct.unpack_from("<H", data, 22)[0], 1)  # CRC flag
        self.assertEqual(data[24], 1)                                # f32
        self.assertEqual(data[25], 0)                                # row-major
        self.assertEqual(struct.unpack_from("<I", data, 28)[0], 3)  # layers
        self.assertEqual(struct.unpack_from("<I", data, 40)[0], 3)  # name length
        self.assertEqual(data[44:47], b"abc")

    def test_size_is_exactly_what_the_records_add_up_to(self):
        layers = small_model()
        body = sum(20 + 4 * l.weight.size + 4 + (4 * l.out_features if l.bias is not None else 0)
                   for l in layers)
        self.assertEqual(len(cfdnn.to_bytes(layers, "abc")), 40 + 4 + 3 + body + 4)

    def test_weights_are_stored_row_major_out_by_in_without_transpose(self):
        w = np.arange(6, dtype=np.float32).reshape(2, 3)  # [out=2][in=3]
        data = cfdnn.to_bytes([Dense(w)])
        first = 40 + 4 + 20
        self.assertEqual(np.frombuffer(data[first:first + 24], "<f4").tolist(),
                         [0, 1, 2, 3, 4, 5])

    def test_zlib_crc_is_the_c_crc(self):
        data = cfdnn.to_bytes(small_model(), "crc")
        self.assertEqual(struct.unpack("<I", data[-4:])[0], crc32_like_c(data[:-4]))

    def test_round_trip_is_exact(self):
        layers = small_model()
        back, name = cfdnn.from_bytes(cfdnn.to_bytes(layers, "rt"))
        self.assertEqual(name, "rt")
        for a, b in zip(layers, back):
            np.testing.assert_array_equal(a.weight, b.weight)
            self.assertEqual(a.activation, b.activation)
            self.assertEqual(np.float32(a.act_param), np.float32(b.act_param))
            if a.bias is None:
                self.assertIsNone(b.bias)
            else:
                np.testing.assert_array_equal(a.bias, b.bias)


class RefusalTests(unittest.TestCase):
    """Each refusal carries the status the C reader returns for the same bytes."""

    def setUp(self):
        self.data = bytearray(cfdnn.to_bytes(small_model(), "x"))

    def test_bit_rot_in_a_weight_is_an_io_error(self):
        # A weight byte: structurally still valid, so only the CRC can catch it.
        # (A flipped size field is caught earlier, as INVALID, by both readers.)
        self.data[40 + 4 + 1 + 20 + 5] ^= 0x01
        self.assertEqual(status_of(lambda: cfdnn.from_bytes(bytes(self.data))), "CFD_ERROR_IO")

    def test_truncation_is_an_io_error(self):
        self.assertEqual(status_of(lambda: cfdnn.from_bytes(bytes(self.data[:-9]))),
                         "CFD_ERROR_IO")

    def test_bad_magic_is_invalid(self):
        self.data[0] = ord("X")
        self.assertEqual(status_of(lambda: cfdnn.from_bytes(bytes(self.data))),
                         "CFD_ERROR_INVALID")

    def test_unknown_version_flag_layout_and_reserved_are_unsupported(self):
        for offset, value in ((8, 2), (22, 3), (25, 1), (32, 1)):
            data = bytearray(self.data)
            data[offset] = value
            self.assertEqual(status_of(lambda: cfdnn.from_bytes(bytes(data))),
                             "CFD_ERROR_UNSUPPORTED", f"offset {offset}")

    def test_writer_refuses_mismatched_shapes(self):
        bad = [Dense(np.zeros((4, 3))), Dense(np.zeros((1, 5)))]
        self.assertEqual(status_of(lambda: cfdnn.to_bytes(bad)), "CFD_ERROR_INVALID")

    def test_writer_refuses_non_finite_weights(self):
        w = np.zeros((1, 3))
        w[0, 1] = np.nan
        self.assertEqual(status_of(lambda: cfdnn.to_bytes([Dense(w)])), "CFD_ERROR_INVALID")

    def test_writer_refuses_an_overlong_name_rather_than_truncating(self):
        self.assertEqual(status_of(lambda: cfdnn.to_bytes(small_model(), "n" * 4097)),
                         "CFD_ERROR_INVALID")


class FoldingTests(unittest.TestCase):
    def test_input_normalization_fold_is_exact(self):
        layers = small_model()
        mean, std = np.array([1.0, -2.0, 0.5]), np.array([2.0, 0.25, 3.0])
        folded = cfdnn.fold_input_normalization(layers, mean, std)
        x = np.random.default_rng(2).normal(size=(64, 3)) * 3.0
        np.testing.assert_allclose(cfdnn.predict(folded, x),
                                   cfdnn.predict(layers, (x - mean) / std), rtol=2e-5, atol=2e-6)

    def test_batchnorm_fold_is_exact(self):
        rng = np.random.default_rng(3)
        dense = Dense(rng.normal(size=(4, 3)), rng.normal(size=4), "relu")
        gamma, beta = rng.normal(size=4), rng.normal(size=4)
        mu, var = rng.normal(size=4), rng.random(4) + 0.5
        folded = cfdnn.fold_batchnorm(dense, gamma, beta, mu, var, 1e-5)

        x = rng.normal(size=(32, 3))
        z = x @ dense.weight.astype(np.float64).T + np.asarray(dense.bias)
        expect = np.maximum(gamma * (z - mu) / np.sqrt(var + 1e-5) + beta, 0.0)
        np.testing.assert_allclose(cfdnn.predict([folded], x), expect, rtol=1e-5, atol=1e-5)


# --- stand-ins for torch modules; the adapter duck-types on class name -----

class Linear:
    def __init__(self, w, b=None):
        self.weight, self.bias = np.asarray(w, np.float32), b


class BatchNorm1d:
    def __init__(self, n, rng):
        self.num_features, self.eps = n, 1e-5
        self.weight, self.bias = rng.normal(size=n), rng.normal(size=n)
        self.running_mean, self.running_var = rng.normal(size=n), rng.random(n) + 0.5


class ReLU:
    pass


class Dropout:
    pass


class LeakyReLU:
    negative_slope = 0.2


class Conv1d:
    pass


class Seq(list):
    training = False


class TorchAdapterTests(unittest.TestCase):
    def test_sequential_with_batchnorm_and_dropout_converts(self):
        rng = np.random.default_rng(4)
        l1 = Linear(rng.normal(size=(5, 3)), rng.normal(size=5))
        bn = BatchNorm1d(5, rng)
        l2 = Linear(rng.normal(size=(1, 5)))
        layers = cfdnn.from_torch_sequential(Seq([l1, bn, ReLU(), Dropout(), l2, LeakyReLU()]))

        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].activation, cfdnn.ACT_RELU)
        self.assertEqual(layers[1].activation, cfdnn.ACT_LEAKY_RELU)
        self.assertAlmostEqual(layers[1].act_param, 0.2)

        x = rng.normal(size=(16, 3))
        z = x @ l1.weight.astype(np.float64).T + np.asarray(l1.bias)
        z = bn.weight * (z - bn.running_mean) / np.sqrt(bn.running_var + bn.eps) + bn.bias
        z = np.maximum(z, 0.0) @ l2.weight.astype(np.float64).T
        z = np.where(z < 0, 0.2 * z, z)
        np.testing.assert_allclose(cfdnn.predict(layers, x), z, rtol=1e-5, atol=1e-5)

    def test_unknown_module_is_refused_not_skipped(self):
        seq = Seq([Linear(np.zeros((2, 3))), Conv1d()])
        self.assertEqual(status_of(lambda: cfdnn.from_torch_sequential(seq)),
                         "CFD_ERROR_INVALID")

    def test_training_mode_is_refused(self):
        seq = Seq([Linear(np.zeros((2, 3)))])
        seq.training = True
        self.assertEqual(status_of(lambda: cfdnn.from_torch_sequential(seq)),
                         "CFD_ERROR_INVALID")

    def test_a_child_left_in_training_mode_is_refused(self):
        # eval() on the parent, then one child flipped back: the parent flag
        # alone would pass, and the export would fold running stats the live
        # model is not using, or drop a Dropout that is still masking.
        rng = np.random.default_rng(6)
        for child in (BatchNorm1d(2, rng), Dropout()):
            child.training = True
            seq = Seq([Linear(rng.normal(size=(2, 3))), child, ReLU()])
            self.assertEqual(status_of(lambda: cfdnn.from_torch_sequential(seq)),
                             "CFD_ERROR_INVALID", type(child).__name__)

    def test_batchnorm_after_activation_cannot_be_folded(self):
        rng = np.random.default_rng(5)
        seq = Seq([Linear(rng.normal(size=(2, 3))), ReLU(), BatchNorm1d(2, rng)])
        self.assertEqual(status_of(lambda: cfdnn.from_torch_sequential(seq)),
                         "CFD_ERROR_INVALID")


class DistilledModelTests(unittest.TestCase):
    """The committed C header must still be what the committed script produces."""

    HEADER = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "nn",
                          "cfdnn_python_golden.h")

    def test_header_bytes_parse_and_match_the_embedded_expectations(self):
        if not os.path.exists(self.HEADER):
            self.skipTest("golden header not generated")
        text = open(self.HEADER, encoding="utf-8").read()
        body = text.split("k_python_golden[")[1].split("};")[0].split("{", 1)[1]
        data = bytes(int(t, 16) for t in body.replace(",", " ").split())
        layers, name = cfdnn.from_bytes(data)
        self.assertEqual(name, "beta-s-star-distilled")

        probes = text.split("k_python_golden_probes[")[1].split("};")[0].split("{", 1)[1]
        rows = [[float(v) for v in r.strip(" {},\n").split(",")]
                for r in probes.split("}") if r.strip(" ,\n")]
        expect = text.split("k_python_golden_expect[")[1].split("};")[0].split("{", 1)[1]
        expect = [float(v) for v in expect.replace("\n", " ").split(",") if v.strip()]
        np.testing.assert_allclose(cfdnn.predict(layers, np.array(rows))[:, 0], expect,
                                   rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
