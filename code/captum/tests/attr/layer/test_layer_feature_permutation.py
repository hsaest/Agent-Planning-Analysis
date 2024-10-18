# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import torch
from captum.attr._core.layer.layer_feature_permutation import LayerFeaturePermutation
from tests.helpers import BaseTest
from tests.helpers.basic import assertTensorAlmostEqual
from tests.helpers.basic_models import BasicModel_MultiLayer
from torch import Tensor


class TestLayerFeaturePermutation(BaseTest):
    def test_single_input(self) -> None:
        net = BasicModel_MultiLayer()
        feature_importance = LayerFeaturePermutation(
            forward_func=net,
            layer=net.linear0,
        )

        batch_size = 2
        input_size = (3,)
        constant_value = 10000

        inp = torch.randn((batch_size,) + input_size)
        inp[:, 0] = constant_value

        attribs = feature_importance.attribute(inputs=inp)

        self.assertTrue(isinstance(attribs, Tensor))
        self.assertEqual(len(attribs), 4)
        self.assertEqual(attribs.squeeze(0).size(), (2 * batch_size,) + input_size)
        zeros = torch.zeros(2 * batch_size)
        assertTensorAlmostEqual(self, attribs[:, 0], zeros, delta=0, mode="max")
        self.assertTrue((attribs[:, 1 : input_size[0]].abs() > 0).all())
