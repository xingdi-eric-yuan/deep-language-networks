from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import numpy as np
import copy
import itertools

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

    def __init__(self, init, forward_template, forward_evaluate):
        self.prompt = init
        self.forward_template = forward_template
        self.forward_evaluate = forward_evaluate

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
        trainable: bool = True,
        **kwargs,
    ):
        forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.node = Node(init, forward_template, forward_evaluate)
        self.prompt_sampler = prompt_sampler
        self.input_sampler = input_sampler
        self.scorer = scorer
        self.trainable = trainable

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def prompt(self):
        return self.node.prompt

    def _update_prompt(self, prompt):
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def prompt_print(self):
        return self.node.prompt

    def backward(self, task, backward_info, is_first_layer=False):
        inputs = [item.input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        previous_prompt = copy.copy(self.prompt)
        if self.trainable:
            # update \pi
            # 1) sample \pi proposals
            pi_candidates = self.prompt_sampler(task, backward_info)
            # 2) rank the candidates
            best_prompt = self.scorer.get_best_prompt(pi_candidates, inputs, gt_outputs)
            # 3) update prompt with the best candidate
            self._update_prompt(best_prompt)

        # update inputs
        if is_first_layer:
            return previous_prompt, self.prompt, inputs, inputs
        new_inputs = []
        for i in range(len(backward_info)):
            # 1) sample input proposals
            input_candidates = self.input_sampler(self.prompt, backward_info[i])  # num_samples
            # 2) rank the inputs
            best_input = self.scorer.get_best_input(self.prompt, input_candidates, gt_outputs[i])
            # 3) collect new inputs
            new_inputs.append(best_input)
        return previous_prompt, self.prompt, inputs, new_inputs


class WideLayer(BaseLayer):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        prompt_sampler: "Sampler",
        input_sampler: "Sampler",
        scorer: "Scorer",
        init: list = None,
        aggregation: str = "concat",
        trainable: bool = True
    ):
        forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.width = len(init)
        self.node_list: Node = [
            Node(i, forward_template, forward_evaluate) for i in init
        ]
        self.forward_evaluate = forward_evaluate
        self.prompt_sampler = prompt_sampler
        self.input_sampler = input_sampler
        self.scorer = scorer
        self.trainable = trainable
        self.aggregation = aggregation
        assert self.aggregation in ["concat", "summary"]
        self.aggregation_forward_template = load_template(
            "aggr_" + self.aggregation + "_forward",
            template_directory="./templates"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def prompt(self):
        return [_n.prompt for _n in self.node_list]

    def prompt_print(self):
        output = ""
        for i in range(self.width):
            output += "--------------------------------\n"
            output += f" - Node {i}: {self.node_list[i].prompt}\n"
        return output

    def _update_prompt(self, prompts):
        assert isinstance(prompts, list) and len(prompts) == len(self.node_list)
        for i in range(len(prompts)):
            self.node_list[i].update_prompt(prompts[i])

    def aggregate(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        if self.aggregation == "concat":
            outputs = []
            for j in range(len(inputs[0])):  # batch size
                batch_inputs = [inputs[i][j] for i in range(len(inputs))]
                outputs.append(self.aggregation_forward_template.render(inputs=batch_inputs))
            return outputs
        elif self.aggregation == "summary":
            llm_inputs = []
            for j in range(len(inputs[0])):  # batch size
                batch_inputs = [inputs[i][j] for i in range(len(inputs))]
                llm_inputs.append(self.aggregation_forward_template.render(inputs=batch_inputs))
            summary = self.forward_evaluate(llm_inputs,
                                            async_generation=True,
                                            **kwargs,)
            return summary
        else:
            raise NotImplementedError

    def forward(self, inputs: Iterable[str], **kwargs) -> np.asarray:
        outputs = [node(inputs, **kwargs) for node in self.node_list]
        outputs = self.aggregate(outputs, **kwargs)
        return np.asarray(outputs)

    def backward(self, task, backward_info, is_first_layer=False):
        inputs = [item.input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        previous_prompt = copy.copy(self.prompt)

        if self.aggregation == "concat":
            # update prompts
            if self.trainable:
                best_prompt_list = []
                for _ in range(len(self.node_list)):
                    # update \pi for each node independently, each of them aims to maximize the logprob of the target
                    # 1) sample \pi proposals
                    pi_candidates = self.prompt_sampler(task, backward_info)
                    # 2) rank the candidates
                    best_prompt = self.scorer.get_best_prompt(pi_candidates, inputs, gt_outputs)
                    # 3) put best prompt into the list
                    best_prompt_list.append(best_prompt)
                # 4) update prompt all together
                self._update_prompt(best_prompt_list)

            # update inputs
            if is_first_layer:
                return previous_prompt, self.prompt, inputs, inputs
            new_inputs = []
            for i in range(len(backward_info)):
                # 1) sample input proposals
                input_candidates = self.input_sampler(self.prompt, backward_info[i])  # num_prompts x num_samples
                # 2) rank the inputs
                best_input = self.scorer.get_best_input4WideConcat(self.prompt, input_candidates, gt_outputs[i])
                # 3) collect new inputs
                new_inputs.append(best_input)
            return previous_prompt, self.prompt, inputs, new_inputs

        elif self.aggregation == "summary":
            # update prompts
            if self.trainable:
                prompt_candidate_list = []
                for _ in range(len(self.node_list)):
                    # update \pi jointly, together they aim to maximize the logprob of the target
                    # 1) sample \pi proposals
                    pi_candidates = self.prompt_sampler(task, backward_info)
                    prompt_candidate_list.append(pi_candidates)
                # 2) rank the candidate tuples
                best_prompt = self.scorer.get_best_prompt4WideSummary(prompt_candidate_list, inputs, gt_outputs, self.aggregation_forward_template)
                # 3) update prompt all together
                self._update_prompt(list(best_prompt))
            # update inputs
            if is_first_layer:
                return previous_prompt, self.prompt, inputs, inputs

            new_inputs = []
            for i in range(len(backward_info)):
                # 1) sample input proposals
                input_candidates = self.input_sampler(self.prompt, backward_info[i])  # num_samples
                # 2) rank the inputs
                best_input = self.scorer.get_best_input4WideSummary(self.prompt, input_candidates, gt_outputs[i], self.aggregation_forward_template)
                # 3) collect new inputs
                new_inputs.append(best_input)
            return previous_prompt, self.prompt, inputs, new_inputs
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"Layer({repr(self.node_list)})"


class DLN_1(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task

        prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=num_samples)
        input_sampler = InputSampler(self.backward_evaluate, "ln_input_backward", num_samples=num_samples)  # HiddenSampler hidden_backward
        scorer_final_layer = LogProbsScorer(self.forward_evaluate, "ln_forward_final_layer")

        self.l1 = LanguageLayer(
            forward_evaluate,
            "ln_forward_final_layer",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer_final_layer,
            init="Let's think step by step.",
            trainable=True,
        )
        self.inputs, self.outputs = [], []
    
    def zero_grad(self):
        self.inputs, self.outputs = [], []

    def save_model(self):
        return self.l1.node.prompt
    
    def load_model(self, init):
        assert isinstance(init, str)
        self.l1._update_prompt(init)

    def forward(self, x):
        # x: batch of strings
        self.inputs = ["\n".join([self.task, _x]) for _x in x]
        self.outputs = self.l1(self.inputs)  # batch
        return self.outputs

    def backward(self, gt):
        # gt: batch of strings
        # l1
        l1_backward_info = [LNBackwardInfo(_i, _o, _gt) for _i, _o, _gt in zip(self.inputs, self.outputs, gt)]
        _ = self.l1.backward(self.task, l1_backward_info, is_first_layer=True)


class DLN_2(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task

        prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=num_samples)
        input_sampler = InputSampler(self.backward_evaluate, "ln_input_backward", num_samples=num_samples)  # HiddenSampler hidden_backward
        scorer_final_layer = LogProbsScorer(self.forward_evaluate, "ln_forward_final_layer")
        scorer = LogProbsScorer(self.forward_evaluate, "ln_forward")

        self.l1 = LanguageLayer(
            forward_evaluate,
            "ln_forward",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer,
            init="Let's think step by step.",
            trainable=True,
        )
        self.l2 = LanguageLayer(
            forward_evaluate,
            "ln_forward_final_layer",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer_final_layer,
            init="Therefore, the answer is:",
            trainable=True,
        )
        self.inputs, self.h, self.outputs = [], [], []
    
    def zero_grad(self):
        self.inputs, self.h, self.outputs = [], [], []

    def save_model(self):
        return [self.l1.node.prompt, self.l2.node.prompt]
    
    def load_model(self, init):
        assert isinstance(init, list) and len(init) == 2
        self.l1._update_prompt(init[0])
        self.l2._update_prompt(init[1])

    def forward(self, x):
        # x: batch of strings
        self.inputs = ["\n".join([self.task, _x]) for _x in x]
        self.h = self.l1(self.inputs)  # batch
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


class DWLN_2(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5, aggregation="concat", width=2):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task
        self.aggregation = aggregation
        self.width = width

        if self.aggregation == "concat":
            wide_layer_prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=num_samples)
            wide_layer_input_sampler = InputSampler4WideConcat(self.backward_evaluate, "ln_input_backward", num_samples=num_samples)
        elif self.aggregation == "summary":
            wide_layer_prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=num_samples)
            wide_layer_input_sampler = InputSampler(self.backward_evaluate, "ln_input_backward", num_samples=num_samples)
        else:
            raise NotImplementedError
        prompt_sampler = PromptSampler(self.backward_evaluate, "ln_prompt_backward", num_samples=num_samples)
        input_sampler = InputSampler(self.backward_evaluate, "ln_input_backward", num_samples=num_samples)  # HiddenSampler hidden_backward
        scorer_final_layer = LogProbsScorer(self.forward_evaluate, "ln_forward_final_layer")
        scorer = LogProbsScorer(self.forward_evaluate, "ln_forward")
        _init_list = ["Let's think step by step."] * self.width

        self.l1 = WideLayer(
            forward_evaluate,
            "ln_forward",
            prompt_sampler=wide_layer_prompt_sampler,
            input_sampler=wide_layer_input_sampler,
            scorer=scorer,
            init=_init_list,
            aggregation=self.aggregation,
            trainable=True,
        )
        self.l2 = LanguageLayer(
            forward_evaluate,
            "ln_forward_final_layer",
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer_final_layer,
            init="Therefore, the answer is:",
            trainable=True,
        )
        self.zero_grad()
    
    def zero_grad(self):
        self.inputs, self.h, self.outputs = [], [], []

    def save_model(self):
        return [item.prompt for item in self.l1.node_list] + [self.l2.node.prompt]
    
    def load_model(self, init):
        assert isinstance(init, list) and len(init) == self.l1.width + 1
        for i in range(self.l1.width):
            self.l1.node_list[i].update_prompt(init[i])
        self.l2._update_prompt(init[-1])

    def forward(self, x):
        # x: batch of strings
        self.inputs = ["\n".join([self.task, _x]) for _x in x]
        self.h = self.l1(self.inputs)  # batch
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


class InputSampler4WideConcat(Sampler):

    def sample(self, prompt_list, backward_info, **kwargs):
        """ Sample new inputs using the backward template.
            Returns a numpy array of shape (self.num_samples * len(prompt_list))
        """
        outputs = []
        for prompt in prompt_list:
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
            outputs += sampled_inputs
        return np.asarray(outputs)


class Scorer(ABC):

    def __init__(self, forward_evaluate, forward_template, eval_kwargs=None):
        self.forward_evaluate = forward_evaluate
        self.forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.eval_kwargs = eval_kwargs or {}
        self.forward_kwargs = {
            "temperature": 0,
            "max_tokens": 512,
        }


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

    def _forward_unique_evals(self, eval_batch, forward=False):
        # there might be doubles in the eval_batch, so we need to only perform unique evals
        unique_keys = list(set(eval_batch))
        unique_keys_to_positions = {key: i for i, key in enumerate(unique_keys)}
        unique_eval_results = self.forward_evaluate(
            unique_keys,
            async_generation=True,
            **self.eval_kwargs if forward is False else self.forward_kwargs,
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
    
    def get_best_input(self, prompt, inputs, gt_output, **kwargs):
        contexts = self._render_context([prompt], inputs)[0]  # inputs
        eval_batch = [f"{contexts[j]}\n{gt_output}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)  # inputs
        best_indexes = logprobs_results.logp_targets.argmax(axis=-1)  # 1
        best_input = inputs[best_indexes]
        return best_input

    def get_best_input4WideConcat(self, prompt_list, inputs, gt_output, **kwargs):
        # Select the best input candidate that maximizes the logprob of the target, summed over all prompts
        num_prompts, num_inputs = len(prompt_list), len(inputs)
        contexts = self._render_context(prompt_list, inputs)  # num_prompts x inputs
        eval_batch = []
        for i in range(num_prompts):
            eval_batch += [f"{contexts[i][j]}\n{gt_output}" for j in range(num_inputs)]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)
        scores = logprobs_results.logp_targets.reshape(
            num_prompts, num_inputs
        ).sum(axis=0)  # num_inputs
        best_indexes = scores.argmax(axis=-1)  # 1
        best_input = inputs[best_indexes]
        return best_input
    
    def get_best_input4WideSummary(self, prompt_list, inputs, gt_output, aggregation_forward_template, **kwargs):
        # Select the best input candidate that maximizes the logprob of the target
        num_nodes, num_inputs = len(prompt_list), len(inputs)
        # get the contexts for each node and each input candidate
        contexts = self._render_context(prompt_list, inputs)  # num_prompts x inputs
        # call the LLM for each node separately (first call)
        h_list = []
        for i in range(num_nodes):
            tmp = self._forward_unique_evals(contexts[i], forward=True)  # inputs
            h_list.append(tmp)  # num_nodes x inputs
        # now aggregate the h_list
        eval_batch = []
        contexts = []
        for i in range(num_inputs):
            _inputs = [h_list[j][i] for j in range(num_nodes)]  # num_nodes
            _inputs = aggregation_forward_template.render(inputs=_inputs)
            contexts.append(_inputs)
            eval_batch.append(f"{_inputs}\n{gt_output}")

        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)  # inputs

        best_indexes = logprobs_results.logp_targets.argmax(axis=-1)  # 1
        best_input = inputs[best_indexes]
        return best_input

    def get_best_prompt4WideSummary(self, prompts_candidates, inputs, gt_outputs, aggregation_forward_template, **kwargs):
        # prompts_candidates: num_nodes x num_candidates per node
        # inputs: num_inputs
        # gt_outputs: num_inputs
        num_nodes = len(prompts_candidates)
        num_candidates_per_node = len(prompts_candidates[0])
        num_inputs = len(inputs)  # also the number of gt_outputs
        # get the contexts for each node
        h_list = []
        for i in range(num_nodes):
            _first_step_context = self._render_context([prompts_candidates[i][j] for j in range(num_candidates_per_node)], inputs)  # num_candidates_per_node x inputs
            tmp = []
            for j in range(num_candidates_per_node):
                tmp += _first_step_context[j]  # inputs
            # tmp: num_candidates*inputs
            h_list.append(self._forward_unique_evals(tmp, forward=True))
        # h_list: num_nodes x num_candidates_per_node*inputs
        # now aggregate the h_list
        eval_batch = []
        gt_outputs_expand = gt_outputs * num_candidates_per_node
        contexts = []
        for i in range(len(h_list[0])):  # num_candidates_per_node*inputs
            _inputs = [h_list[j][i] for j in range(num_nodes)]
            _inputs = aggregation_forward_template.render(inputs=_inputs)
            contexts.append(_inputs)
            eval_batch.append(f"{_inputs}\n{gt_outputs_expand[i]}")
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)  # num_candidates*inputs

        scores = logprobs_results.logp_targets.reshape(
            num_candidates_per_node, num_inputs
        ).sum(axis=-1)  # num_candidates
        best_indexes = scores.argmax(axis=-1)  # 1
        best_prompt = [item[best_indexes] for item in prompts_candidates]  # num_nodes
        return best_prompt
