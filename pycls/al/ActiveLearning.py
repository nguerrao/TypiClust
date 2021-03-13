# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------

import numpy as np 
import torch
from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu
import os

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using args.sampling_method.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet

        NOTE: args is obtained in class property 
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty_uniform_discretize":
            old_train_mode = clf_model.training
            old_penultimate_mode = clf_model.penultimate_active
            clf_model.eval()
            
            activeSet, uSet = self.sampler.uncertainty_uniform_discretize(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, lSet=lSet, \
                uSet=uSet, model=clf_model, dataset=trainDataset)
            
            clf_model.train(old_train_mode)
            clf_model.penultimate_active = old_penultimate_mode

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "centre_of_gravity" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN =="cog":
            wastrain = clf_model.training
            clf_model.eval()
            waslatent = clf_model.penultimate_active
            clf_model.penultimate_active = True
            activeSet, uSet = self.sampler.centre_of_gravity(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet\
                ,model=clf_model,dataset=trainDataset,istopK=True)
            clf_model.train(wastrain)
            clf_model.penultimate_active = waslatent

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset = trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model,dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_dbal":
            activeSet, uSet = self.sampler.ensemble_dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels,dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_bald":
            activeSet, uSet = self.sampler.ensemble_bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels,dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels,dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPPLING_FN == "vaal":
            None
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
