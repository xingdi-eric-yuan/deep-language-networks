from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import numpy as np
import copy

from dln.operator import LLM
from dln.template import load_template


@dataclass
class LogProbs:
    logp_targets: np.ndarray
    distribution: np.ndarray


@dataclass
class LNBackwardInfo:
    input: str = None
    output: str = None
    target: str = None


class Node(ABC):

    def __init__(self, init, forward_template, forward_evaluate, layer):
        self.prompt = init
        self.forward_template = forward_template
        self.forward_evaluate = forward_evaluate
        self.layer = layer

    def update_prompt(self, prompt):
        self.prompt = prompt

    def render_template(self, x):
        # x: batch x input_len
        tpl_inputs = [
            self.forward_template.render(input=i, prompt=self.prompt)
            for i in x
        ]
        return tpl_inputs

    def __call__(self, x, **kwargs):
        tpl_inputs = self.render_template(x)
        fwd_outputs = self.forward_evaluate(
            tpl_inputs,
            stop=self.forward_template.stop_tokens,
            **kwargs,
        )
        return np.asarray(fwd_outputs)


class BaseLayer(ABC):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        prompt_sampler: "Sampler",
        input_sampler: "Sampler",
        scorer: "Scorer",
        init: str = None,
        **kwargs,
    ):
        forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.nodes = Node(init, forward_template, forward_evaluate)
        self.prompt_sampler = prompt_sampler
        self.input_sampler = input_sampler
        self.scorer = scorer

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def prompt(self):
        return self.node.prompt

    def _update_prompts(self, prompt):
        self.node.update_prompt(prompt)

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        outputs = self.node(inputs, **kwargs)
        return np.asarray(outputs)

    def backward(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Layer({repr(self.node)})"


class LanguageLayer(BaseLayer):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def backward(self, task, backward_info, is_first_layer=False):

        inputs = [item.input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        # update \pi
        # 1) sample \pi proposals
        pi_candidates = self.prompt_sampler(task, backward_info)
        # 2) rank the candidates
        best_prompt = self.scorer.get_best_prompt(pi_candidates, inputs, gt_outputs)
        # 3) update prompt with the best candidate
        previous_prompt = copy.copy(self.prompt)
        self._update_prompts(best_prompt)

        # update inputs
        if is_first_layer:
            return previous_prompt, self.prompt, inputs, inputs
        new_inputs = []
        for _info in backward_info:
            # 1) sample input proposals
            input_candidates = self.input_sampler(self.prompt, _info)  # num_samples
            # 2) rank the inputs
            best_input = self.scorer.get_best_input(self.prompt, input_candidates, gt_outputs)
            # 3) collect new inputs
            new_inputs.append(best_input)
        return previous_prompt, self.prompt, inputs, new_inputs


class DLN_2(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task

        prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=4)
        input_sampler = InputSampler(self.backward_evaluate, "ln_input_backward", num_samples=4)  # HiddenSampler hidden_backward
        scorer = LogProbsScorer(self.forward_evaluate, "ln_forward")

        self.l1 = LanguageLayer(
            forward_evaluate,
            "ln_forward",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer,
            init="Let's think step by step.",
        )
        self.l2 = LanguageLayer(
            forward_evaluate,
            "ln_forward",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer,
            init="Therefore, the answer is:",
        )
        self.inputs, self.h, self.outputs = [], [], []
    
    def zero_grad(self):
        self.inputs, self.h, self.outputs = [], [], []

    def forward(self, x):
        # x: batch of strings
        self.inputs = x
        self.h = self.l1(x)  # batch
        self.outputs = self.l2(self.h)  # batch
        return self.outputs

    def backward(self, gt):
        # gt: batch of strings
        # l2
        l2_backward_info = [LNBackwardInfo(_i, _o, _gt) for _i, _o, _gt in zip(self.h, self.outputs, gt)]
        _, _, _, new_h = self.l2.backward(self.task, l2_backward_info, is_first_layer=False)
        # l1
        l1_backward_info = [LNBackwardInfo(_i, _o, _gt) for _i, _o, _gt in zip(self.inputs, self.h, new_h)]
        _ = self.l1.backward(self.task, l1_backward_info, is_first_layer=True)
        self.zero_grad()


class Sampler(ABC):

    def __init__(self, backward_evaluate, backward_template, num_samples=4):
        self.backward_evaluate = backward_evaluate
        self.backward_template = load_template(
            backward_template,
            template_directory="./templates"
        )
        self.num_samples = num_samples

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class PromptSampler(Sampler):

    def sample(self, task, backward_info, **kwargs):
        """ Sample new prompts using the backward template.
            Returns a numpy array of shape (self.num_samples)
        """
        tpl_inputs = []
        for _ in range(self.num_samples):
            tpl_inputs.append(
                self.backward_template.render(
                    task=task, backward_info=backward_info)
            )

        new_prompts = self.backward_evaluate(
            tpl_inputs,
            stop=self.backward_template.stop_tokens,
            **kwargs,
        )
        return np.asarray(new_prompts)


class InputSampler(Sampler):

    def sample(self, prompt, backward_info, **kwargs):
        """ Sample new inputs using the backward template.
            Returns a numpy array of shape (self.num_samples)
        """
        tpl_inputs = []
        for _ in range(self.num_samples):
            tpl_inputs.append(
                self.backward_template.render(
                    prompt=prompt, target=backward_info.target, output=backward_info.output)
            )

        sampled_inputs = self.backward_evaluate(
            tpl_inputs,
            stop=self.backward_template.stop_tokens,
            **kwargs,
        )
        return np.asarray(sampled_inputs)


class Scorer(ABC):

    def __init__(self, forward_evaluate, forward_template, eval_kwargs=None):
        self.forward_evaluate = forward_evaluate
        self.forward_template = forward_template
        self.eval_kwargs = eval_kwargs or {}

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    @abstractmethod
    def score(self, *args, **kwargs):
        pass


class LogProbsScorer(Scorer):

    def __init__(self, forward_evaluate, forward_template, eval_kwargs=None):
        eval_kwargs = {
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "return_logprobs": True,
            "raw_logprobs": True,
        }
        super().__init__(forward_evaluate, forward_template, eval_kwargs)

    def _render_context(self, prompts, inputs):
        rendered_template = []
        for _p in prompts:
            rendered_template_per_prompt = []
            for _i in inputs:
                fwd_rendered = self.forward_template.render(input=_i, prompt=_p)
                rendered_template_per_prompt.append(fwd_rendered.replace('[END]', ''))  # TODO: clean prompt when generating, not here
            rendered_template.append(rendered_template_per_prompt)
        return rendered_template  # prompts x inputs

    def _forward_unique_evals(self, eval_batch):
        # there might be doubles in the eval_batch, so we need to only perform unique evals
        unique_keys = list(set(eval_batch))
        unique_keys_to_positions = {key: i for i, key in enumerate(unique_keys)}
        unique_eval_results = self.forward_evaluate(
            unique_keys,
            async_generation=True,
            **self.eval_kwargs,
        )
        # get the results in the same order as the eval_batch
        eval_results = []
        for eval_key in eval_batch:
            eval_results.append(unique_eval_results[unique_keys_to_positions[eval_key]])
        return eval_results

    def _get_logprobs_results(self, contexts, eval_results):
        log_probs = [e[1] for e in eval_results]
        output_logprobs = []
        context_logprobs = []

        if isinstance(contexts[0], list):
            contexts_flatten = []
            for i in range(len(contexts)):
                for j in range(len(contexts[i])):
                    contexts_flatten.append(contexts[i][j])
        else:
            contexts_flatten = contexts

        burn_in = 0
        for context, token_log_probs in zip(contexts_flatten, log_probs):
            num_tokens_prompt = len(self.forward_evaluate.encoder.encode(context))
            target_log_probs = token_log_probs[num_tokens_prompt + burn_in:]
            context_log_probs = token_log_probs[1:num_tokens_prompt]

            if len(target_log_probs) == 0:
                output_logprobs.append("empty")
            else:
                output_logprobs.append(
                    sum(target_log_probs) / (len(target_log_probs) + 1e-5)
                )

            context_logprobs.append(
                sum(context_log_probs) / (len(context_log_probs) + 1e-5)
            )

        non_empty = [o for o in output_logprobs if o != "empty"]
        if len(non_empty) == 0:
            min = 0
        else:
            min = np.min(non_empty)
        output_logprobs = [o if o != "empty" else min for o in output_logprobs]
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))  # TODO: reshape?

    def get_best_prompt(self, prompts_candidates, inputs, gt_outputs, **kwargs):
        num_candidates = len(prompts_candidates)
        contexts = self._render_context(prompts_candidates, inputs)  # prompts_candidates x inputs
        eval_batch = []
        for i in range(num_candidates):
            eval_batch += [f"{contexts[i][j]}\n{gt_outputs[j]}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)
        scores = logprobs_results.logp_targets.reshape(
            num_candidates, len(inputs)
        ).sum(axis=-1)  # num_candidates
        best_indexes = scores.argmax(axis=-1)  # 1
        best_prompt = prompts_candidates[best_indexes]
        return best_prompt
    
    def get_best_input(self, prompt, inputs, gt_outputs, **kwargs):
        contexts = self._render_context([prompt], inputs)[0]  # inputs
        eval_batch = []
        eval_batch = [f"{contexts[j]}\n{gt_outputs[j]}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)  # inputs
        best_indexes = logprobs_results.argmax(axis=-1)  # 1
        best_input = inputs[best_indexes]
        return best_input