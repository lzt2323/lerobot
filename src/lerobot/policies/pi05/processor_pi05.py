#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_LABELS,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_pi05 import PI05Config

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


PI05_SUBTASK_PROMPT = "pi05_subtask_prompt"
PI05_SUBTASK_TARGET = "pi05_subtask_target"
PI05_ACTION_PROMPT_PREFIX = "pi05_action_prompt_prefix"
PI05_ACTION_PROMPT_SUFFIX = "pi05_action_prompt_suffix"


def _as_text_list(value: str | list[str] | tuple[str, ...], *, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"{key} must be a string or list of strings, got {type(value)}")


def _clean_prompt_text(text: str) -> str:
    return text.strip().replace("_", " ").replace("\n", " ")


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    subtask_key: str = "subtask"
    enable_subtask_prediction: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        tasks = _as_text_list(tasks, key=self.task_key)

        subtasks = None
        if self.enable_subtask_prediction:
            subtasks_value = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.subtask_key)
            if subtasks_value is None:
                if transition.get(TransitionKey.ACTION) is not None:
                    raise ValueError(
                        "PI05 subtask prediction is enabled, but no subtask was found in complementary data"
                    )
            else:
                subtasks = _as_text_list(subtasks_value, key=self.subtask_key)
                if len(subtasks) != len(tasks):
                    raise ValueError(f"Expected {len(tasks)} subtasks, got {len(subtasks)}")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        subtask_prompts = []
        subtask_targets = []
        action_prompt_prefixes = []
        action_prompt_suffixes = []
        for i, task in enumerate(tasks):
            cleaned_text = _clean_prompt_text(task)
            state_str = " ".join(map(str, discretized_states[i]))
            if self.enable_subtask_prediction:
                subtask_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nSubtask: ")
                cleaned_subtask = _clean_prompt_text(subtasks[i]) if subtasks is not None else ""
                if subtasks is not None:
                    subtask_targets.append(cleaned_subtask)
                action_prompt_prefix = f"Task: {cleaned_text}, Subtask: "
                action_prompt_suffix = f", State: {state_str};\nAction: "
                full_prompt = f"{action_prompt_prefix}{cleaned_subtask}{action_prompt_suffix}"
                action_prompt_prefixes.append(action_prompt_prefix)
                action_prompt_suffixes.append(action_prompt_suffix)
            else:
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        if self.enable_subtask_prediction:
            transition[TransitionKey.COMPLEMENTARY_DATA][PI05_SUBTASK_PROMPT] = subtask_prompts
            if subtasks is not None:
                transition[TransitionKey.COMPLEMENTARY_DATA][PI05_SUBTASK_TARGET] = subtask_targets
            transition[TransitionKey.COMPLEMENTARY_DATA][PI05_ACTION_PROMPT_PREFIX] = action_prompt_prefixes
            transition[TransitionKey.COMPLEMENTARY_DATA][PI05_ACTION_PROMPT_SUFFIX] = action_prompt_suffixes
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi05_tokenizer_processor_step")
@dataclass
class Pi05TokenizerProcessorStep(ProcessorStep):
    tokenizer_name: str
    max_length: int = 200
    subtask_max_length: int = 128
    task_key: str = "task"
    enable_subtask_prediction: bool = False
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    input_tokenizer: Any = None

    def __post_init__(self):
        if not _transformers_available or AutoTokenizer is None:
            raise ImportError("transformers is required to use Pi05TokenizerProcessorStep")
        self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.input_tokenizer.padding_side = self.padding_side

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}

        tasks = _as_text_list(complementary_data.get(self.task_key), key=self.task_key)
        tokenized_prompt = self._tokenize_text(tasks, max_length=self.max_length)
        observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        if self.enable_subtask_prediction:
            prompts = _as_text_list(complementary_data.get(PI05_SUBTASK_PROMPT), key=PI05_SUBTASK_PROMPT)
            targets_value = complementary_data.get(PI05_SUBTASK_TARGET)
            if targets_value is None:
                tokenized_subtask = self._tokenize_text(prompts, max_length=self.subtask_max_length)
                subtask_tokens = tokenized_subtask["input_ids"]
                subtask_masks = tokenized_subtask["attention_mask"].to(dtype=torch.bool)
                subtask_labels = None
            else:
                targets = _as_text_list(targets_value, key=PI05_SUBTASK_TARGET)
                if len(prompts) != len(targets):
                    raise ValueError(f"Expected {len(prompts)} subtask targets, got {len(targets)}")
                subtask_tokens, subtask_masks, subtask_labels = self._tokenize_subtask_lm(prompts, targets)
            observation[OBS_LANGUAGE_SUBTASK_TOKENS] = subtask_tokens
            observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = subtask_masks
            if subtask_labels is not None:
                observation[OBS_LANGUAGE_SUBTASK_LABELS] = subtask_labels

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def _tokenize_text(self, text: str | list[str], *, max_length: int) -> dict[str, torch.Tensor]:
        return self.input_tokenizer(
            text,
            max_length=max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )

    def _tokenize_subtask_lm(
        self, prompts: list[str], targets: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eos_token = self.input_tokenizer.eos_token or ""
        full_texts = [f"{prompt}{target}{eos_token}" for prompt, target in zip(prompts, targets, strict=True)]
        tokenized_full = self.input_tokenizer(
            full_texts,
            max_length=self.subtask_max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized_full.pop("offset_mapping")

        labels = tokenized_full["input_ids"].clone()
        labels[tokenized_full["attention_mask"] == 0] = -100
        for row, prompt in enumerate(prompts):
            prompt_len = len(prompt)
            prompt_or_special = offset_mapping[row, :, 1] <= prompt_len
            labels[row, prompt_or_special] = -100

        return (
            tokenized_full["input_ids"],
            tokenized_full["attention_mask"].to(dtype=torch.bool),
            labels,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "subtask_max_length": self.subtask_max_length,
            "task_key": self.task_key,
            "enable_subtask_prediction": self.enable_subtask_prediction,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if self.enable_subtask_prediction:
            for key in (
                OBS_LANGUAGE_SUBTASK_TOKENS,
                OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
                OBS_LANGUAGE_SUBTASK_LABELS,
            ):
                if key not in features[PipelineFeatureType.OBSERVATION]:
                    features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                        type=FeatureType.LANGUAGE, shape=(self.subtask_max_length,)
                    )
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            enable_subtask_prediction=config.enable_subtask_prediction,
        ),
        Pi05TokenizerProcessorStep(
            tokenizer_name=os.environ.get("PI0_TOKENIZER_NAME", "google/paligemma-3b-pt-224"),
            max_length=config.tokenizer_max_length,
            subtask_max_length=config.subtask_tokenizer_max_length,
            enable_subtask_prediction=config.enable_subtask_prediction,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
