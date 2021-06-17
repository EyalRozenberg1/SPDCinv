from abc import ABC
from jax import jit
from typing import Tuple, Sequence, Dict, Any, Union, Optional, List

import jax.numpy as np
import numpy as onp
import itertools
import os
from spdc_inv import DATA_DIR


class Loss(ABC):
    def __init__(
            self,
            observable_as_target: Tuple[Dict[Any, bool], ...],
            target: str = None,
            loss_arr: Union[Dict[Any, Optional[Tuple[str, str]]],
                            Tuple[Dict[Any, Optional[Tuple[str, str]]], ...]] = None,
            loss_weights: Union[Dict[Any, Optional[Tuple[float, float]]],
                                Tuple[Dict[Any, Optional[Tuple[float, float]]], ...]] = None,
            reg_observable: Union[Dict[Any, Optional[Tuple[str, str]]],
                                  Tuple[Dict[Any, Optional[Tuple[str, str]]], ...]] = None,
            reg_observable_w: Union[Dict[Any, Optional[Tuple[float, float]]],
                                    Tuple[Dict[Any, Optional[Tuple[float, float]]], ...]] = None,
            reg_observable_elements: Union[Dict[Any, Optional[Tuple[List[int], List[int]]]],
                                           Tuple[Dict[Any, Optional[Tuple[List[int], List[int]]]], ...]] = None,
            l2_reg: float = 0.,

    ):
        self.LOSS = dict(l1=self.l1,
                         l2=self.l2,
                         kl=self.kl,
                         bhattacharyya=self.bhattacharyya,
                         trace_distance=self.trace_distance)

        self.REG_OBS = dict(sparsify=self.sparsify,
                            equalize=self.equalize)

        if loss_arr is not None:
            assert len(loss_arr) == len(loss_weights), 'loss_arr and loss_weights must have equal number of elements'

            assert observable_as_target and os.path.exists(os.path.join(DATA_DIR, 'targets', target)), \
                f' target file, {target}, is missing'

            for loss_arr_ in loss_arr:
                assert loss_arr_.lower() in self.LOSS, f'Loss must be defined as on of the following' \
                                                  f'options only: {list(self.LOSS.keys())}'

        if reg_observable is not None:
            assert len(reg_observable) == len(reg_observable_w) and \
                   len(reg_observable) == len(reg_observable_elements), 'reg_observable, reg_observable_w and' \
                                                                        'reg_observable_elements must have equal ' \
                                                                        'number of elements'
            for reg_obs in reg_observable:
                assert reg_obs.lower() in self.REG_OBS, f'Loss must be defined as on of the following' \
                                                        f'options only: {list(self.REG_OBS.keys())}'

        self.observable_as_target = observable_as_target
        self.loss_arr = loss_arr
        self.loss_weights = loss_weights
        self.reg_observable = reg_observable
        self.reg_observable_w = reg_observable_w
        self.reg_observable_elements = reg_observable_elements
        self.l2_reg = l2_reg

        self.target_str = None
        if observable_as_target and loss_arr is not None:
            self.target_str = target

        self.loss_stack = self.LOSS_stack()
        self.reg_obs_stack = self.REG_obs_stack()


    def apply(
            self,
            observable,
            model_parameters,
            target,
    ):

        loss = 0.
        if self.observable_as_target:

            for loss_func, loss_weight in zip(self.loss_stack, self.loss_weights):
                loss = loss + loss_weight * loss_func(observable, target)

            for obs_func, weight, elements in zip(self.reg_obs_stack,
                                                  self.reg_observable_w, self.reg_observable_elements):
                loss = loss + weight * obs_func(observable, elements)

        if self.l2_reg > 0.:
            loss = loss + self.l2_reg * self.l2_regularization(model_parameters)

        return loss

    def LOSS_stack(self):

        loss_stack = []
        if self.loss_arr is None:
            self.loss_weights = []
            return loss_stack

        for loss in self.loss_arr:
            loss_stack.append(self.LOSS[loss])

        return loss_stack

    def REG_obs_stack(self):

        reg_obs_stack = []
        if self.reg_observable is None:
            self.reg_observable_w = []
            self.reg_observable_elements = []
            return reg_obs_stack

        for reg_obs in self.reg_observable:
            reg_obs_stack.append(self.REG_OBS[reg_obs])

        return reg_obs_stack

    @staticmethod
    @jit
    def l1(observable: np.array,
           target: np.array,
           ):
        """
        L1 loss
        Parameters
        ----------
        observable: tensor

        Returns the l1 distance between observable and the target
        -------

        """
        return np.sum(np.abs(observable - target))

    @staticmethod
    @jit
    def l2(observable: np.array,
           target: np.array,
           ):
        """
        L2 loss
        Parameters
        ----------
        observable: tensor

        Returns the l2 distance between observable and the target
        -------

        """
        return np.sum((observable - target)**2)

    @staticmethod
    @jit
    def kl(observable: np.array,
           target: np.array,
           eps: float = 1e-2,
           ):
        """
        kullback leibler divergence
        Parameters
        ----------
        observable: tensor
        eps: Epsilon is used here to avoid conditional code for
                checking that neither observable nor the target is equal to 0 (or smaller)

        Returns the kullback leibler divergence between observable and the target
        -------

        """

        A = observable + eps
        B = target + eps
        return np.sum(B * np.log(B / A))

    @staticmethod
    @jit
    def bhattacharyya(observable: np.array,
                      target: np.array,
                      eps: float = 1e-10,
                      ):
        """
        Bhattacharyya distance

        Parameters
        ----------
        observable: tensor
        eps: Epsilon is used here to avoid conditional code for
                checking that neither observable nor the target is equal to 0 (or smaller)

        Returns the Bhattacharyya distance between observable and the target
        -------

        """

        return np.sqrt(1. - np.sum(np.sqrt(observable * target + eps)))

    @staticmethod
    @jit
    def trace_distance(rho: np.array,
                       target: np.array,
                       ):
        """
        Trace Distance

        Calculate the Trace Distance between rho and the target density matrix
        as depict in: https://en.wikipedia.org/wiki/Trace_distance#Definition

        Parameters
        ----------
        rho: density matrix rho
        Returns: Trace distance between rho and the target
        -------
        """

        td = 0.5 * np.linalg.norm(rho - target, ord='nuc')

        return td

    @staticmethod
    def sparsify(observable,
                 elements,
                 ):
        """
        the method will penalize all other elements in tensor observable
        Parameters
        ----------
        observable: tensor of size (1,projection_n_modes)
        elements: elements we don't want to penalize

        Returns l1 amplitude of all other elements
        -------

        """
        projection_n_modes = observable.shape[-1]
        sparsify_elements = onp.delete(onp.arange(projection_n_modes ** 2), elements)
        return np.sum(np.abs(observable[..., sparsify_elements]))

    @staticmethod
    def equalize(
            observable,
            elements
    ):
        """
        the method will penalize if elements in observable doesn't have equal amplitude
        Parameters
        ----------
        observable: tensor of size (1,projection_n_modes)
        elements: elements we wish to have equal energy in observable

        Returns the sum over all l1 distances fro all elements in observable
        -------

        """
        equalize_elements_combinations = list(itertools.combinations(elements, 2))
        reg = 0.
        for el_comb in equalize_elements_combinations:
            reg = reg + np.sum(np.abs(observable[..., el_comb[0]] - observable[..., el_comb[1]]))

        return reg

    @staticmethod
    @jit
    def l2_regularization(model_parameters):
        """
        l2 regularization
        Parameters
        ----------
        model_parameters: model's learned parameters

        Returns l2 regularization
        -------

        """
        l2_reg = 0.
        for params in model_parameters:
            if params is not None:
                l2_reg = l2_reg + np.sum(np.abs(params)**2)
        return l2_reg
