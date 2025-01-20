# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import os
import random
import string
import tempfile
import unittest
import warnings
from copy import deepcopy
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch
import torch.multiprocessing
from parameterized import parameterized

from monai import config
from monai.data import DataLoader, Dataset
from monai.data.meta_obj import get_track_meta, set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import decollate_batch, list_data_collate
from monai.transforms import BorderPadd, Compose, DivisiblePadd, FromMetaTensord, ToMetaTensord
from monai.utils.enums import PostFix
from monai.utils.module import pytorch_after
from tests.util import TEST_DEVICES, SkipIfBeforePyTorchVersion, assert_allclose, skip_if_no_cuda

DTYPES = [[torch.float32], [torch.float64], [torch.float16], [torch.int64], [torch.int32], [None]]
TESTS = []
for _device in TEST_DEVICES:
    for _dtype in DTYPES:
        TESTS.append((*_device, *_dtype))  # type: ignore


def rand_string(min_len=5, max_len=10):
    str_size = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.punctuation
    return "".join(random.choice(chars) for _ in range(str_size))


class TestMetaTensor(unittest.TestCase):
    @staticmethod
    def get_im(shape=None, dtype=None, device=None):
        if shape is None:
            shape = (1, 10, 8)
        affine = torch.randint(0, 10, (4, 4))
        meta = {"fname": rand_string()}
        t = torch.rand(shape)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        m = MetaTensor(t.clone(), affine, meta)
        return m, t

    def check_ids(self, a, b, should_match):
        comp = self.assertEqual if should_match else self.assertNotEqual
        comp(id(a), id(b))

    def check_meta(self, a: MetaTensor, b: MetaTensor) -> None:
        self.assertEqual(a.is_batch, b.is_batch)
        meta_a, meta_b = a.meta, b.meta
        # need to split affine from rest of metadata
        aff_a = meta_a.get("affine", None)
        aff_b = meta_b.get("affine", None)
        assert_allclose(aff_a, aff_b)
        meta_a = {k: v for k, v in meta_a.items() if k != "affine"}
        meta_b = {k: v for k, v in meta_b.items() if k != "affine"}
        self.assertEqual(meta_a, meta_b)

    def check(
        self,
        out: torch.Tensor,
        orig: torch.Tensor,
        *,
        shape: bool = True,
        vals: bool = True,
        ids: bool = True,
        device: str | torch.device | None = None,
        meta: bool = True,
        check_ids: bool = True,
        **kwargs,
    ):
        if device is None:
            device = orig.device

        # check the image
        self.assertIsInstance(out, type(orig))
        if shape:
            assert_allclose(torch.as_tensor(out.shape), torch.as_tensor(orig.shape))
        if vals:
            assert_allclose(out, orig, **kwargs)
        if check_ids:
            self.check_ids(out, orig, ids)
        self.assertTrue(str(device) in str(out.device))

        # check meta and affine are equal and affine is on correct device
        if isinstance(orig, MetaTensor) and isinstance(out, MetaTensor) and meta:
            self.check_meta(orig, out)
            if check_ids:
                self.check_ids(out.meta, orig.meta, ids)

    @parameterized.expand(TESTS)
    def test_as_tensor(self, device, dtype):
        m, t = self.get_im(device=device, dtype=dtype)
        t2 = m.as_tensor()
        self.assertIsInstance(t2, torch.Tensor)
        self.assertNotIsInstance(t2, MetaTensor)
        self.assertIsInstance(m, MetaTensor)
        self.check(t, t2, ids=False)

    def test_as_dict(self):
        m, _ = self.get_im()
        m_dict = m.as_dict("im")
        im, meta = m_dict["im"], m_dict[PostFix.meta("im")]
        affine = meta.pop("affine")
        m2 = MetaTensor(im, affine, meta)
        self.check(m2, m, check_ids=False)

    @parameterized.expand(TESTS)
    def test_constructor(self, device, dtype):
        m, t = self.get_im(device=device, dtype=dtype)
        # construct from pre-existing
        m1 = MetaTensor(m.clone())
        self.check(m, m1, ids=False, meta=False)
        # meta already has affine
        m2 = MetaTensor(t.clone(), meta=m.meta)
        self.check(m, m2, ids=False, meta=False)
        # meta dosen't have affine
        affine = m.meta.pop("affine")
        m3 = MetaTensor(t.clone(), affine=affine, meta=m.meta)
        self.check(m, m3, ids=False, meta=False)

    @parameterized.expand(TESTS)
    @skip_if_no_cuda
    def test_to_cuda(self, device, dtype):
        """Test `to`, `cpu` and `cuda`. For `to`, check args and kwargs."""
        orig, _ = self.get_im(device=device, dtype=dtype)
        m = orig.clone()
        m = m.to("cuda")
        self.check(m, orig, ids=False, device="cuda")
        m = m.cpu()
        self.check(m, orig, ids=False, device="cpu")
        m = m.cuda()
        self.check(m, orig, ids=False, device="cuda")
        m = m.to("cpu")
        self.check(m, orig, ids=False, device="cpu")
        m = m.to(device="cuda")
        self.check(m, orig, ids=False, device="cuda")
        m = m.to(device="cpu")
        self.check(m, orig, ids=False, device="cpu")

    @skip_if_no_cuda
    def test_affine_device(self):
        m, _ = self.get_im()  # device="cuda")
        m.affine = torch.eye(4)
        self.assertTrue("cpu" in str(m.affine.device))

    @parameterized.expand(TESTS)
    def test_copy(self, device, dtype):
        m, _ = self.get_im(device=device, dtype=dtype)
        # shallow copy
        a = m
        self.check(a, m, ids=True)
        # deepcopy
        a = deepcopy(m)
        self.check(a, m, ids=False)
        # clone
        a = m.clone(memory_format=torch.preserve_format)
        a = m.clone()
        self.check(a, m, ids=False)
        a = MetaTensor([[]], device=device, dtype=dtype)
        self.check(a, deepcopy(a), ids=False)

    @parameterized.expand(TESTS)
    def test_add(self, device, dtype):
        m1, t1 = self.get_im(device=device, dtype=dtype)
        m2, t2 = self.get_im(device=device, dtype=dtype)
        self.check(m1 + m2, t1 + t2, ids=False)
        self.check(torch.add(m1, m2), t1 + t2, ids=False)
        self.check(torch.add(input=m1, other=m2), t1 + t2, ids=False)
        self.check(torch.add(m1, other=m2), t1 + t2, ids=False)
        m3 = deepcopy(m2)
        t3 = deepcopy(t2)
        m3 += 3
        t3 += 3
        self.check(m3, t3, ids=False)
        # check torch.Tensor+MetaTensor and MetaTensor+torch.Tensor
        self.check(torch.add(m1, t2), t1 + t2, ids=False)
        self.check(torch.add(t2, m1), t1 + t2, ids=False)

    @parameterized.expand(TEST_DEVICES)
    def test_conv(self, device):
        im, _ = self.get_im((1, 3, 10, 8, 12), device=device)
        conv = torch.nn.Conv3d(im.shape[1], 5, 3)
        conv.to(device)
        out = conv(im)
        self.check(out, im, shape=False, vals=False, ids=False)

    @parameterized.expand(TESTS)
    def test_stack(self, device, dtype):
        numel = 3
        ims = [self.get_im(device=device, dtype=dtype)[0] for _ in range(numel)]
        stacked = torch.stack(ims)
        self.assertIsInstance(stacked, MetaTensor)
        orig_affine = ims[0].meta.pop("affine")
        stacked_affine = stacked.meta.pop("affine")
        assert_allclose(orig_affine, stacked_affine)
        self.assertEqual(stacked.meta, ims[0].meta)

    def test_get_set_meta_fns(self):
        set_track_meta(False)
        self.assertFalse(get_track_meta())
        set_track_meta(True)
        self.assertTrue(get_track_meta())

    @parameterized.expand(TEST_DEVICES)
    def test_torchscript(self, device):
        shape = (1, 3, 10, 8)
        im, _ = self.get_im(shape, device=device)
        conv = torch.nn.Conv2d(im.shape[1], 5, 3)
        conv.to(device)
        im_conv = conv(im)
        traced_fn = torch.jit.trace(conv, im.as_tensor())
        # save it, load it, use it
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "im.pt")
            torch.jit.save(traced_fn, f=fname)
            traced_fn = torch.jit.load(fname)
            out = traced_fn(im)
            self.assertIsInstance(out, torch.Tensor)
            if not isinstance(out, MetaTensor) and not pytorch_after(1, 9, 1):
                warnings.warn(
                    "When calling `nn.Module(MetaTensor) on a module traced with "
                    "`torch.jit.trace`, your version of pytorch returns a "
                    "`torch.Tensor` instead of a `MetaTensor`. Consider upgrading "
                    "your pytorch version if this is important to you."
                )
                im_conv = im_conv.as_tensor()
        self.check(out, im_conv, ids=False)

    def test_pickling(self):
        m, _ = self.get_im()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "im.pt")
            torch.save(m, fname)
            m2 = torch.load(fname)
            if not isinstance(m2, MetaTensor) and not pytorch_after(1, 8, 1):
                warnings.warn("Old version of pytorch. pickling converts `MetaTensor` to `torch.Tensor`.")
                m = m.as_tensor()
        self.check(m2, m, ids=False)

    @skip_if_no_cuda
    def test_amp(self):
        shape = (1, 3, 10, 8)
        device = "cuda"
        im, _ = self.get_im(shape, device=device)
        conv = torch.nn.Conv2d(im.shape[1], 5, 3)
        conv.to(device)
        im_conv = conv(im)
        with torch.cuda.amp.autocast():
            im_conv2 = conv(im)
        self.check(im_conv2, im_conv, ids=False, rtol=1e-2, atol=1e-2)

    def test_out(self):
        """Test when `out` is given as an argument."""
        m1, _ = self.get_im()
        m2, _ = self.get_im()
        m3, _ = self.get_im()
        torch.add(m2, m3, out=m1)
        m1_add = m2 + m3

        assert_allclose(m1, m1_add)
        # self.check_meta(m1, m2)  # meta is from first input tensor

    @parameterized.expand(TESTS)
    def test_collate(self, device, dtype):
        numel = 3
        ims = [self.get_im(device=device, dtype=dtype)[0] for _ in range(numel)]
        ims = [MetaTensor(im, applied_operations=[f"t{i}"]) for i, im in enumerate(ims)]
        collated = list_data_collate(ims)
        # tensor
        self.assertIsInstance(collated, MetaTensor)
        expected_shape = (numel,) + tuple(ims[0].shape)
        self.assertTupleEqual(tuple(collated.shape), expected_shape)
        for i, im in enumerate(ims):
            self.check(im, ims[i], ids=True)
        # affine
        self.assertIsInstance(collated.affine, torch.Tensor)
        expected_shape = (numel,) + tuple(ims[0].affine.shape)
        self.assertTupleEqual(tuple(collated.affine.shape), expected_shape)
        self.assertEqual(len(collated.applied_operations), numel)

    @parameterized.expand(TESTS)
    def test_dataset(self, device, dtype):
        ims = [self.get_im(device=device, dtype=dtype)[0] for _ in range(4)]
        ds = Dataset(ims)
        for i, im in enumerate(ds):
            self.check(im, ims[i], ids=True)

    @parameterized.expand(DTYPES)
    @SkipIfBeforePyTorchVersion((1, 8))
    def test_dataloader(self, dtype):
        batch_size = 5
        ims = [self.get_im(dtype=dtype)[0] for _ in range(batch_size * 2)]
        ims = [MetaTensor(im, applied_operations=[f"t{i}"]) for i, im in enumerate(ims)]
        ds = Dataset(ims)
        im_shape = tuple(ims[0].shape)
        affine_shape = tuple(ims[0].affine.shape)
        expected_im_shape = (batch_size,) + im_shape
        expected_affine_shape = (batch_size,) + affine_shape
        dl = DataLoader(ds, num_workers=batch_size, batch_size=batch_size)
        for batch in dl:
            self.assertIsInstance(batch, MetaTensor)
            self.assertTupleEqual(tuple(batch.shape), expected_im_shape)
            self.assertTupleEqual(tuple(batch.affine.shape), expected_affine_shape)
            self.assertEqual(len(batch.applied_operations), batch_size)

    @SkipIfBeforePyTorchVersion((1, 9))
    def test_indexing(self):
        """
        Check the metadata is returned in the expected format depending on whether
        the input `MetaTensor` is a batch of data or not.
        """
        ims = [self.get_im()[0] for _ in range(5)]
        data = list_data_collate(ims)

        # check that when using non-batch data, metadata is copied wholly when indexing
        # or iterating across data.
        im = ims[0]
        self.check_meta(im[0], im)
        self.check_meta(next(iter(im)), im)

        self.assertEqual(im[None].shape, (1, 1, 10, 8))
        self.assertEqual(data[None].shape, (1, 5, 1, 10, 8))

        # index
        d = data[0]
        self.check(d, ims[0], ids=False)

        # iter
        d = next(iter(data))
        self.check(d, ims[0], ids=False)

        # complex indexing

        # `is_batch==True`, should have subset of image and metadata.
        d = data[1:3]
        self.check(d, list_data_collate(ims[1:3]), ids=False)

        # is_batch==True, should have subset of image and same metadata as `[1:3]`.
        d = data[1:3, 0]
        self.check(d, list_data_collate([i[0] for i in ims[1:3]]), ids=False)

        # `is_batch==False`, should have first metadata and subset of first image.
        d = data[0, 0]
        self.check(d, ims[0][0], ids=False)
        self.assertEqual(d.applied_operations, ims[0][0].applied_operations)

        # `is_batch==True`, should have all metadata and subset of all images.
        d = data[:, 0]
        self.check(d, list_data_collate([i[0] for i in ims]), ids=False)

        # `is_batch==True`, should have all metadata and subset of all images.
        d = data[..., -1]
        self.check(d, list_data_collate([i[..., -1] for i in ims]), ids=False)

        # `is_batch==False`, tuple split along batch dim. Should have individual
        # metadata.
        d = data.unbind(0)
        self.assertIsInstance(d, tuple)
        self.assertEqual(len(d), len(ims))
        for _d, _im in zip(d, ims):
            self.check(_d, _im, ids=False)

        # `is_batch==False`, tuple split along batch dim. Should have individual
        # metadata.
        d = data.unbind(dim=0)
        self.assertIsInstance(d, tuple)
        self.assertEqual(len(d), len(ims))
        for _d, _im in zip(d, ims):
            self.check(_d, _im, ids=False)
            self.assertEqual(_d.applied_operations, _im.applied_operations)

        # `is_batch==True`, tuple split along non-batch dim. Should have all metadata.
        d = data.unbind(-1)
        self.assertIsInstance(d, tuple)
        self.assertEqual(len(d), ims[0].shape[-1])
        for _d in d:
            self.check_meta(_d, data)

        # `is_batch==True`, tuple split along non-batch dim. Should have all metadata.
        d = data.unbind(dim=-1)
        self.assertIsInstance(d, tuple)
        self.assertEqual(len(d), ims[0].shape[-1])
        for _d in d:
            self.check_meta(_d, data)

    def test_slicing(self):
        x = MetaTensor(np.zeros((10, 3, 4)))
        self.assertEqual(x[slice(4, 1)].shape[0], 0)
        x.is_batch = True
        with self.assertRaises(ValueError):
            x[slice(0, 8)]
        x = MetaTensor(np.zeros((3, 3, 4)))
        x.is_batch = True
        self.assertEqual(x[torch.tensor([True, False, True])].shape, (2, 3, 4))
        self.assertEqual(x[[True, False, True]].shape, (2, 3, 4))

    @parameterized.expand(DTYPES)
    @SkipIfBeforePyTorchVersion((1, 8))
    def test_decollate(self, dtype):
        batch_size = 3
        ims = [self.get_im(dtype=dtype)[0] for _ in range(batch_size * 2)]
        ds = Dataset(ims)
        dl = DataLoader(ds, num_workers=batch_size, batch_size=batch_size)
        batch = next(iter(dl))
        decollated = decollate_batch(batch)
        self.assertIsInstance(decollated, list)
        self.assertEqual(len(decollated), batch_size)
        for elem, im in zip(decollated, ims):
            self.assertIsInstance(elem, MetaTensor)
            self.check(elem, im, ids=False)

    def test_str(self):
        t = MetaTensor([1.0], affine=torch.tensor(1), meta={"fname": "filename"})
        self.assertEqual(str(t), "metatensor([1.])")
        self.assertEqual(t.__repr__(), "metatensor([1.])")
        self.assertEqual(f"{t[0]:.2f}", "1.00")

    def test_shape(self):
        s = MetaTensor([1])
        self.assertEqual(s.shape, torch.Size([1]))
        self.assertEqual(s.size(), torch.Size([1]))
        self.assertEqual(s.size(0), 1)

    def test_astype(self):
        t = MetaTensor([1.0], affine=torch.tensor(1), meta={"fname": "filename"})
        for np_types in ("float32", "np.float32", "numpy.float32", np.float32, float, "int", np.uint16):
            self.assertIsInstance(t.astype(np_types), np.ndarray)
        for pt_types in ("torch.float", torch.float, "torch.float64"):
            self.assertIsInstance(t.astype(pt_types), torch.Tensor)
        self.assertIsInstance(t.astype("torch.float", device="cpu"), torch.Tensor)

    def test_transforms(self):
        key = "im"
        _, im = self.get_im()
        tr = Compose([ToMetaTensord(key), BorderPadd(key, 1), DivisiblePadd(key, 16), FromMetaTensord(key)])
        num_tr = len(tr.transforms)
        data = {key: im, PostFix.meta(key): {"affine": torch.eye(4)}}

        # apply one at a time
        for i, _tr in enumerate(tr.transforms):
            data = _tr(data)
            is_meta = isinstance(_tr, (ToMetaTensord, BorderPadd, DivisiblePadd))
            if is_meta:
                self.assertEqual(len(data), 1 if not config.USE_META_DICT else 2)  # im, im_transforms, compatibility
                self.assertIsInstance(data[key], MetaTensor)
                n_applied = len(data[key].applied_operations)
            else:
                self.assertEqual(len(data), 3)  # im, im_meta_dict, im_transforms
                self.assertIsInstance(data[key], torch.Tensor)
                self.assertNotIsInstance(data[key], MetaTensor)
                n_applied = len(data[PostFix.transforms(key)])

            self.assertEqual(n_applied, i + 1)

        # inverse one at a time
        for i, _tr in enumerate(tr.transforms[::-1]):
            data = _tr.inverse(data)
            is_meta = isinstance(_tr, (FromMetaTensord, BorderPadd, DivisiblePadd))
            if is_meta:
                self.assertEqual(len(data), 1)  # im
                self.assertIsInstance(data[key], MetaTensor)
                n_applied = len(data[key].applied_operations)
            else:
                self.assertEqual(len(data), 3)  # im, im_meta_dict, im_transforms
                self.assertIsInstance(data[key], torch.Tensor)
                self.assertNotIsInstance(data[key], MetaTensor)
                n_applied = len(data[PostFix.transforms(key)])

            self.assertEqual(n_applied, num_tr - i - 1)

        # apply all in one go
        data = tr({key: im, PostFix.meta(key): {"affine": torch.eye(4)}})
        self.assertEqual(len(data), 3)  # im, im_meta_dict, im_transforms
        self.assertIsInstance(data[key], torch.Tensor)
        self.assertNotIsInstance(data[key], MetaTensor)
        n_applied = len(data[PostFix.transforms(key)])
        self.assertEqual(n_applied, num_tr)

        # inverse all in one go
        data = tr.inverse(data)
        self.assertEqual(len(data), 3)  # im, im_meta_dict, im_transforms
        self.assertIsInstance(data[key], torch.Tensor)
        self.assertNotIsInstance(data[key], MetaTensor)
        n_applied = len(data[PostFix.transforms(key)])
        self.assertEqual(n_applied, 0)

    def test_construct_with_pre_applied_transforms(self):
        key = "im"
        _, im = self.get_im()
        tr = Compose([BorderPadd(key, 1), DivisiblePadd(key, 16)])
        data = tr({key: im})
        m = MetaTensor(im, applied_operations=data["im"].applied_operations)
        self.assertEqual(len(m.applied_operations), len(tr.transforms))

    def test_pending_ops(self):
        m, _ = self.get_im()
        self.assertEqual(m.pending_operations, [])
        self.assertEqual(m.peek_pending_shape(), (10, 8))
        self.assertIsInstance(m.peek_pending_affine(), torch.Tensor)
        self.assertTrue(m.peek_pending_rank() >= 1)
        m.push_pending_operation({})
        self.assertEqual(m.peek_pending_shape(), (10, 8))
        self.assertIsInstance(m.peek_pending_affine(), torch.Tensor)
        self.assertTrue(m.peek_pending_rank() >= 1)

    @parameterized.expand(TESTS)
    def test_multiprocessing(self, device=None, dtype=None):
        """multiprocessing sharing with 'device' and 'dtype'"""
        buf = io.BytesIO()
        t = MetaTensor([0, 0] if dtype in (torch.int32, torch.int64) else [0.0, 0.0], device=device, dtype=dtype)
        t.is_batch = True
        if t.is_cuda:
            with self.assertRaises(NotImplementedError):
                ForkingPickler(buf).dump(t)
            return
        ForkingPickler(buf).dump(t)
        obj = ForkingPickler.loads(buf.getvalue())
        self.assertIsInstance(obj, MetaTensor)
        assert_allclose(obj.as_tensor(), t)
        assert_allclose(obj.is_batch, True)

    @parameterized.expand(TESTS)
    def test_array_function(self, device="cpu", dtype=float):
        a = np.random.RandomState().randn(100, 100)
        b = MetaTensor(a, device=device)
        assert_allclose(np.sum(a), np.sum(b))
        assert_allclose(np.sum(a, axis=1), np.sum(b, axis=1))
        assert_allclose(np.linalg.qr(a), np.linalg.qr(b))
        c = MetaTensor(
            [1, 2, 3] if dtype in (torch.int32, torch.int64) else [1.0, 2.0, 3.0], device=device, dtype=dtype
        )
        assert_allclose(np.argwhere(c == 1.0).astype(int).tolist(), [[0]])
        assert_allclose(np.concatenate([c, c]), np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
        if pytorch_after(1, 8, 1):
            assert_allclose(c > np.asarray([1.0, 1.0, 1.0]), np.asarray([False, True, True]))
            assert_allclose(
                c > torch.as_tensor([1.0, 1.0, 1.0], device=device), torch.as_tensor([False, True, True], device=device)
            )

    @parameterized.expand(TESTS)
    def test_numpy(self, device=None, dtype=None):
        """device, dtype"""
        t = MetaTensor([0 if dtype in (torch.int32, torch.int64) else 0.0], device=device, dtype=dtype)
        self.assertIsInstance(t, MetaTensor)
        assert_allclose(t.array, np.asarray([0.0]))
        t.array = np.asarray([1.0])
        self.check_meta(t, MetaTensor([1.0]))
        assert_allclose(t.as_tensor(), torch.as_tensor([1.0]))
        t.array = [2.0]
        self.check_meta(t, MetaTensor([2.0]))
        assert_allclose(t.as_tensor(), torch.as_tensor([2.0]))
        if not t.is_cuda:
            t.array[0] = torch.as_tensor(3 if dtype in (torch.int32, torch.int64) else 3.0, device=device, dtype=dtype)
            self.check_meta(t, MetaTensor([3.0]))
            assert_allclose(t.as_tensor(), torch.as_tensor([3.0]))


if __name__ == "__main__":
    unittest.main()
