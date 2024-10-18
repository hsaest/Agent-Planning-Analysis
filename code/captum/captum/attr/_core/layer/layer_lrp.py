#!/usr/bin/env python3

# pyre-strict
import typing
from typing import Any, cast, List, Tuple, Union

from captum._utils.common import (
    _format_tensor_into_tuples,
    _reduce_list,
    _sort_key_list,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    undo_gradient_requirements,
)
from captum._utils.typing import (
    Literal,
    ModuleOrModuleList,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.lrp import LRP
from captum.attr._utils.attribution import LayerAttribution
from torch import Tensor
from torch.nn import Module


class LayerLRP(LRP, LayerAttribution):
    r"""
    Layer-wise relevance propagation is based on a backward propagation
    mechanism applied sequentially to all layers of the model. Here, the
    model output score represents the initial relevance which is decomposed
    into values for each neuron of the underlying layers. The decomposition
    is defined by rules that are chosen for each layer, involving its weights
    and activations. Details on the model can be found in the original paper
    [https://doi.org/10.1371/journal.pone.0130140]. The implementation is
    inspired by the tutorial of the same group
    [https://doi.org/10.1016/j.dsp.2017.10.011] and the publication by
    Ancona et al. [https://openreview.net/forum?id=Sy21R9JAW].
    """

    def __init__(self, model: Module, layer: ModuleOrModuleList) -> None:
        """
        Args:

            model (Module): The forward function of the model or
                        any modification of it. Custom rules for a given layer need to
                        be defined as attribute
                        `module.rule` and need to be of type PropagationRule.

            layer (torch.nn.Module or list(torch.nn.Module)): Layer or layers
                          for which attributions are computed.
                          The size and dimensionality of the attributions
                          corresponds to the size and dimensionality of the layer's
                          input or output depending on whether we attribute to the
                          inputs or outputs of the layer. If value is None, the
                          relevance for all layers is returned in attribution.
        """
        LayerAttribution.__init__(self, model, layer)
        LRP.__init__(self, model)
        if hasattr(self.model, "device_ids"):
            # pyre-fixme[4]: Attribute must be annotated.
            self.device_ids = cast(List[int], self.model.device_ids)

    @typing.overload  # type: ignore
    # pyre-fixme[43]: The implementation of `attribute` does not accept all possible
    #  arguments of overload defined on line `66`.
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        additional_forward_args: Any = None,
        # pyre-fixme[9]: return_convergence_delta has type `Literal[]`; used as `bool`.
        # pyre-fixme[31]: Expression `Literal[False]` is not a valid type.
        # pyre-fixme[24]: Non-generic type `typing.Literal` cannot take parameters.
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]: ...

    @typing.overload
    # pyre-fixme[43]: The implementation of `attribute` does not accept all possible
    #  arguments of overload defined on line `77`.
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        *,
        # pyre-fixme[31]: Expression `Literal[True]` is not a valid type.
        # pyre-fixme[24]: Non-generic type `typing.Literal` cannot take parameters.
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Tuple[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Union[Tensor, List[Tensor]],
    ]: ...

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, ...],
        List[Union[Tensor, Tuple[Tensor, ...]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Union[Tensor, List[Tensor]],
        ],
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which relevance is
                        propagated.
                        If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, Tensor, or list, optional): Output indices for
                    which gradients are computed (for classification cases,
                    this is usually the target class).
                    If the network returns a scalar value per example,
                    no target index is necessary.
                    For general 2D outputs, targets can be either:

                    - a single integer or a tensor containing a single
                        integer, which is applied to all input examples

                    - a list of integers or a 1D tensor, with length matching
                        the number of examples in inputs (dim 0). Each integer
                        is applied as the target for the corresponding example.

                    For outputs with > 2 dimensions, targets can be either:

                    - A single tuple, which contains #output_dims - 1
                        elements. This target index is applied to all examples.

                    - A list of tuples with length equal to the number of
                        examples in inputs (dim 0), and each tuple containing
                        #output_dims - 1 elements. Each tuple is applied as the
                        target for the corresponding example.

                    Default: None
            additional_forward_args (tuple, optional): If the forward function
                    requires additional arguments other than the inputs for
                    which attributions should not be computed, this argument
                    can be provided. It must be either a single additional
                    argument of a Tensor or arbitrary (non-tuple) type or a tuple
                    containing multiple additional arguments including tensors
                    or any arbitrary python types. These arguments are provided to
                    model in order, following the arguments in inputs.
                    Note that attributions are not computed with respect
                    to these arguments.
                    Default: None

            return_convergence_delta (bool, optional): Indicates whether to return
                    convergence delta or not. If `return_convergence_delta`
                    is set to True convergence delta will be returned in
                    a tuple following attributions.
                    Default: False

            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.

            verbose (bool, optional): Indicates whether information on application
                    of rules is printed during propagation.
                    Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions** or 2-element tuple of
            **attributions**, **delta** or list of **attributions** and **delta**:

              - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The propagated relevance values with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations. If attributions for all layers are returned
                        (layer=None) a list of tensors or tuples of tensors is returned
                        with entries for each layer.
              - **delta** (*Tensor* or list of *Tensor*
                        returned if return_convergence_delta=True):
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        examples in input.
                        If attributions for all layers are returned (layer=None) a list
                        of tensors is returned with entries for
                        each layer.

        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> layer_lrp = LayerLRP(net, net.conv1)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = layer_lrp.attribute(input, target=5)

        """
        # pyre-fixme[16]: `LayerLRP` has no attribute `verbose`.
        self.verbose = verbose
        # pyre-fixme[16]: `LayerLRP` has no attribute `_original_state_dict`.
        self._original_state_dict = self.model.state_dict()
        # pyre-fixme[16]: `LayerLRP` has no attribute `layers`.
        self.layers = []
        self._get_layers(self.model)
        self._check_and_attach_rules()
        # pyre-fixme[16]: `LayerLRP` has no attribute `attribute_to_layer_input`.
        self.attribute_to_layer_input = attribute_to_layer_input
        # pyre-fixme[16]: `LayerLRP` has no attribute `backward_handles`.
        self.backward_handles = []
        # pyre-fixme[16]: `LayerLRP` has no attribute `forward_handles`.
        self.forward_handles = []

        # pyre-fixme[9]: inputs has type `TensorOrTupleOfTensorsGeneric`; used as
        #  `Tuple[Tensor, ...]`.
        inputs = _format_tensor_into_tuples(inputs)
        # pyre-fixme[6]: For 1st argument expected `Tuple[Tensor, ...]` but got
        #  `TensorOrTupleOfTensorsGeneric`.
        gradient_mask = apply_gradient_requirements(inputs)

        try:
            # 1. Forward pass
            output = self._compute_output_and_change_weights(
                # pyre-fixme[6]: For 1st argument expected `Tuple[Tensor, ...]` but
                #  got `TensorOrTupleOfTensorsGeneric`.
                inputs,
                target,
                additional_forward_args,
            )
            self._register_forward_hooks()
            # 2. Forward pass + backward pass
            _ = compute_gradients(
                self._forward_fn_wrapper, inputs, target, additional_forward_args
            )
            relevances = self._get_output_relevance(output)
        finally:
            self._restore_model()
        # pyre-fixme[6]: For 1st argument expected `Tuple[Tensor, ...]` but got
        #  `TensorOrTupleOfTensorsGeneric`.
        undo_gradient_requirements(inputs, gradient_mask)

        if return_convergence_delta:
            delta: Union[Tensor, List[Tensor]]
            if isinstance(self.layer, list):
                delta = []
                for relevance_layer in relevances:
                    delta.append(
                        self.compute_convergence_delta(relevance_layer, output)
                    )
            else:
                delta = self.compute_convergence_delta(
                    cast(Tuple[Tensor, ...], relevances), output
                )
            return relevances, delta  # type: ignore
        else:
            return relevances  # type: ignore

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _get_single_output_relevance(self, layer, output):
        # pyre-fixme[16]: `LayerLRP` has no attribute `attribute_to_layer_input`.
        if self.attribute_to_layer_input:
            normalized_relevances = layer.rule.relevance_input
        else:
            normalized_relevances = layer.rule.relevance_output
        key_list = _sort_key_list(list(normalized_relevances.keys()), self.device_ids)
        normalized_relevances = _reduce_list(
            [normalized_relevances[device_id] for device_id in key_list]
        )

        if isinstance(normalized_relevances, tuple):
            return tuple(
                normalized_relevance
                * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
                for normalized_relevance in normalized_relevances
            )
        else:
            return normalized_relevances * output.reshape(
                (-1,) + (1,) * (normalized_relevances.dim() - 1)
            )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _get_output_relevance(self, output):
        if isinstance(self.layer, list):
            relevances = []
            for layer in self.layer:
                relevances.append(self._get_single_output_relevance(layer, output))
            return relevances
        else:
            return self._get_single_output_relevance(self.layer, output)

    @staticmethod
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _convert_list_to_tuple(
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        relevances: Union[List[Any], Tuple[Any, ...]]
    ) -> Tuple[Any, ...]:
        if isinstance(relevances, list):
            return tuple(relevances)
        else:
            return relevances