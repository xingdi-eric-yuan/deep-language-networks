from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import numpy as np
import copy
import re
import itertools

from dln.operator import LLM
from dln.template import load_template
from dln.loss import NumberPresenceLoss


@dataclass
class LogProbs:
    logp_targets: np.ndarray
    distribution: np.ndarray


@dataclass
class LNBackwardInfo:
    first_step_input: str = None
    input: str = None
    output: str = None
    target: str = None
    loss: float = None


class Node(ABC):

    def __init__(self, init, forward_template, forward_evaluate):
        self.prompt = init
        self.forward_template = forward_template
        self.forward_evaluate = forward_evaluate

    def update_prompt(self, prompt):
        self.prompt = prompt

    def render_template(self, x, x0=None):
        # x: batch x input_len
        # x_minus_one: batch x input_len, or None
        if x0 is None:
            tpl_inputs = [
                self.forward_template.render(input=i, prompt=self.prompt)
                for i in x
            ]
        else:
            assert len(x) == len(x0)  # batch size
            tpl_inputs = [
                self.forward_template.render(input=i, prompt=self.prompt, first_step_input=j)
                for i, j in zip(x, x0)
            ]
        return tpl_inputs

    def __call__(self, x, x0=None, **kwargs):
        tpl_inputs = self.render_template(x, x0=x0)
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
        parent_layer: "BaseLayer" = None,
        contrastive: bool = False,
        score_input_phx: bool = False,
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
        self.parent_layer = parent_layer
        self.contrastive = contrastive
        self.score_input_phx = score_input_phx

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def prompt(self):
        return self.node.prompt

    def _update_prompt(self, prompt):
        self.node.update_prompt(prompt)

    def forward(self, inputs: Iterable[str], first_step_input=None, **kwargs) -> np.asarray:
        outputs = self.node(inputs, first_step_input, **kwargs)
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

    def backward(self, task, backward_info, normalize_score=False, skip_good_h=False, is_first_layer=False):
        inputs = [item.input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        previous_prompt = copy.copy(self.prompt)
        new_inputs = copy.copy(inputs)

        # update inputs
        if not is_first_layer:
            new_inputs = []
            for i in range(len(backward_info)):
                if skip_good_h and backward_info[i].loss == 0:
                    new_inputs.append(inputs[i])
                    continue

                # 1) sample input proposals
                input_candidates = self.input_sampler(self.prompt, backward_info[i])  # num_samples
                # 2) rank the inputs
                best_input = self.scorer.get_best_input(self.prompt, input_candidates, gt_outputs[i],
                                                        backward_info[i].first_step_input, 
                                                        self.parent_layer.prompt if self.parent_layer is not None else None, 
                                                        normalize=normalize_score,
                                                        phx=self.score_input_phx)
                # 3) collect new inputs
                new_inputs.append(best_input)
                backward_info[i].input = best_input

        if self.trainable:
            # update \pi
            # 1) sample \pi proposals
            pi_candidates = self.prompt_sampler(task, self.prompt, backward_info, two_step_sample=is_first_layer)
            pi_candidates = np.concatenate([pi_candidates, np.asarray([self.prompt])])  # add the current prompt
            # 2) rank the candidates
            best_prompt = self.scorer.get_best_prompt(pi_candidates, backward_info, contrastive=is_first_layer and self.contrastive, normalize=normalize_score)
            # 3) update prompt with the best candidate
            self._update_prompt(best_prompt)
        
        return previous_prompt, self.prompt, inputs, new_inputs


class WideLayer(BaseLayer):

    def __init__(
        self,
        forward_evaluate: LLM,
        forward_template: str,
        prompt_sampler: "Sampler",
        scorer: "Scorer",
        init: list = None,
        aggregation: str = "concat",
        trainable: bool = True,
        contrastive: bool = False,
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
        self.scorer = scorer
        self.trainable = trainable
        self.aggregation = aggregation
        self.contrastive = contrastive
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
        output = []
        for i in range(self.width):
            output.append("--------------------------------")
            output.append(f" - Node {i}: {self.node_list[i].prompt}")
        return "\n".join(output)

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

    def backward(self, task, backward_info, normalize_score=False):
        inputs = [item.input for item in backward_info]
        previous_prompt = copy.copy(self.prompt)

        if not self.trainable:
            return previous_prompt, self.prompt, inputs, inputs

        if self.aggregation == "concat":
            # update prompts
            # update \pi for each node independently, each of them aims to maximize the logprob of the target
            pi_candidates_list = []
            for i in range(self.width):
                # 1) sample \pi proposals
                _pi_candidates = self.prompt_sampler(task, self.prompt[i], backward_info)
                pi_candidates_list = pi_candidates_list + _pi_candidates.tolist()
            pi_candidates_list = pi_candidates_list + self.prompt  # add the current prompt
            pi_candidates_list = list(set(pi_candidates_list))
            if len(pi_candidates_list) < self.width:
                pi_candidates_list = pi_candidates_list + self.prompt
            pi_candidates = np.asarray(pi_candidates_list)
            # 2) rank the candidates, take top k, where k is num_nodes
            best_k_prompt = self.scorer.get_best_k_prompt(pi_candidates, backward_info, self.width, contrastive=self.contrastive, normalize=normalize_score)
            # 3) update the top k prompts
            self._update_prompt(best_k_prompt)
        elif self.aggregation == "summary":
            # update prompts
            # update \pi jointly, together they aim to maximize the logprob of the target
            pi_candidates_list = []
            for i in range(self.width):
                # 1) sample \pi proposals
                _pi_candidates = self.prompt_sampler(task, self.prompt[i], backward_info)
                _pi_candidates = np.concatenate([_pi_candidates, np.asarray([self.prompt[i]])])  # add the current prompt
                pi_candidates_list.append(_pi_candidates)
            # 2) rank the candidate tuples
            best_prompt = self.scorer.get_best_prompt4WideSummary(pi_candidates_list, backward_info, self.aggregation_forward_template, contrastive=self.contrastive, normalize=normalize_score)
            # 3) update prompt all together
            self._update_prompt(list(best_prompt))
        else:
            raise NotImplementedError
        return previous_prompt, self.prompt, inputs, inputs

    def __repr__(self):
        return f"Layer({repr(self.node_list)})"


class DLN_1(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5,
                 prompt_backward_template="ln_prompt_backward:1.0", input_backward_template="ln_input_backward:1.0", normalize_score=False):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task
        self.loss_function = NumberPresenceLoss()
        self.normalize_score = normalize_score

        prompt_sampler = PromptSampler(self.backward_evaluate, prompt_backward_template, num_samples=num_samples)
        input_sampler = InputSampler(self.backward_evaluate, input_backward_template, num_samples=num_samples)  # HiddenSampler hidden_backward
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
        self.zero_grad()
    
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
        # loss
        losses = self.loss_function(self.outputs, gt)
        # l1
        l1_backward_info = [LNBackwardInfo(_i0, _i, _o, _gt, _loss) for _i0, _i, _o, _gt, _loss in zip(self.inputs, self.inputs, self.outputs, gt, losses)]
        _ = self.l1.backward(self.task, l1_backward_info, normalize_score=self.normalize_score, is_first_layer=True)


class DLN_2(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5, 
                 prompt_backward_template="ln_prompt_backward:1.0", input_backward_template="ln_input_backward:1.0",
                 first_layer_contrastive=False, score_input_phx=False, normalize_score=False, skip_good_h=False,
                 normalize_by_length=True, two_step_h_sample=False, two_step_pi_sample=False, residual=False):
        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task
        self.loss_function = NumberPresenceLoss()
        self.first_layer_contrastive = first_layer_contrastive
        self.score_input_phx = score_input_phx
        self.normalize_score = normalize_score
        self.skip_good_h = skip_good_h
        self.normalize_by_length = normalize_by_length
        self.two_step_h_sample = two_step_h_sample
        self.two_step_pi_sample = two_step_pi_sample
        self.residual = residual

        two_step_h_sample_template = "two_step_h_sample_template:1.0" if self.two_step_h_sample else None
        two_step_pi_sample_template = "two_step_pi_sample_template:1.0" if self.two_step_pi_sample else None

        l1_template = "ln_forward"
        if self.residual:
            l2_template = "ln_forward_final_layer_residual"
        else:
            l2_template = "ln_forward_final_layer"
        prompt_sampler = PromptSampler(self.backward_evaluate, prompt_backward_template, num_samples=num_samples, two_step_sample_template=two_step_pi_sample_template)
        input_sampler = InputSampler(self.backward_evaluate, input_backward_template, num_samples=num_samples, two_step_sample_template=two_step_h_sample_template)
        scorer_final_layer = LogProbsScorer(self.forward_evaluate, l2_template, l1_template, self.normalize_by_length)
        scorer = LogProbsScorer(self.forward_evaluate, l1_template, None, self.normalize_by_length)

        self.l1 = LanguageLayer(
            forward_evaluate,
            l1_template,
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer,
            init="Let's think step by step.",
            trainable=True,
            contrastive=self.first_layer_contrastive,
            score_input_phx=self.score_input_phx,
        )
        self.l2 = LanguageLayer(
            forward_evaluate,
            l2_template,
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer_final_layer,
            init="Therefore, the answer is:",
            trainable=True,
            parent_layer=self.l1,
            score_input_phx=self.score_input_phx,
        )
        self.zero_grad()
    
    def zero_grad(self):
        self.inputs, self.h, self.new_h, self.outputs = [], [], [], []

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
        self.outputs = self.l2(self.h, self.inputs if self.residual else None)  # batch
        return self.outputs

    def backward(self, gt):
        # gt: batch of strings
        # loss
        losses = self.loss_function(self.outputs, gt)
        # l2
        l2_backward_info = [LNBackwardInfo(_i0, _i, _o, _gt, _loss) for _i0, _i, _o, _gt, _loss in zip(self.inputs, self.h, self.outputs, gt, losses)]
        _, _, _, new_h = self.l2.backward(self.task, l2_backward_info, normalize_score=self.normalize_score, is_first_layer=False, skip_good_h=self.skip_good_h)
        self.new_h = new_h
        # l1
        l1_backward_info = [LNBackwardInfo(_i0, _i, _o, _gt, _loss) for _i0, _i, _o, _gt, _loss in zip(self.inputs, self.inputs, self.h, new_h, losses)]
        _ = self.l1.backward(self.task, l1_backward_info, normalize_score=self.normalize_score, is_first_layer=True)


class DWLN_2(ABC):

    def __init__(self, task, forward_evaluate, backward_evaluate, num_samples=5, aggregation="concat", width=2, 
                 prompt_backward_template="ln_prompt_backward:1.0", input_backward_template="ln_input_backward:1.0",
                 first_layer_contrastive=False, score_input_phx=False, normalize_score=False, skip_good_h=False,
                 normalize_by_length=True, two_step_h_sample=False, two_step_pi_sample=False, residual=False):

        self.forward_evaluate = forward_evaluate
        self.backward_evaluate = backward_evaluate
        self.task = task
        self.aggregation = aggregation
        self.width = width
        self.loss_function = NumberPresenceLoss()
        self.first_layer_contrastive = first_layer_contrastive
        self.score_input_phx = score_input_phx
        self.normalize_score = normalize_score
        self.skip_good_h = skip_good_h
        self.normalize_by_length = normalize_by_length
        self.two_step_h_sample = two_step_h_sample
        self.two_step_pi_sample = two_step_pi_sample
        self.residual = residual

        two_step_h_sample_template = "two_step_h_sample_template:1.0" if self.two_step_h_sample else None
        two_step_pi_sample_template = "two_step_pi_sample_template:1.0" if self.two_step_pi_sample else None

        l1_template = "ln_forward"
        if self.residual:
            l2_template = "ln_forward_final_layer_residual"
        else:
            l2_template = "ln_forward_final_layer"
        prompt_sampler = PromptSampler(self.backward_evaluate, prompt_backward_template, num_samples=num_samples, two_step_sample_template=two_step_pi_sample_template)
        input_sampler = InputSampler(self.backward_evaluate, input_backward_template, num_samples=num_samples, two_step_sample_template=two_step_h_sample_template)
        scorer_final_layer = LogProbsScorer(self.forward_evaluate, l2_template, l1_template, self.normalize_by_length)
        scorer = LogProbsScorer(self.forward_evaluate, l1_template, None, self.normalize_by_length)
        _init_list = ["Let's think step by step."] * self.width

        self.l1 = WideLayer(
            forward_evaluate,
            l1_template,
            prompt_sampler=prompt_sampler,
            scorer=scorer,
            init=_init_list,
            aggregation=self.aggregation,
            trainable=True,
            contrastive=self.first_layer_contrastive,
            score_input_phx=self.score_input_phx,
        )
        self.l2 = LanguageLayer(
            forward_evaluate,
            l2_template,
            prompt_sampler=prompt_sampler,
            input_sampler=input_sampler,
            scorer=scorer_final_layer,
            init="Therefore, the answer is:",
            trainable=True,
            parent_layer=self.l1,
            score_input_phx=self.score_input_phx,
        )
        self.zero_grad()
    
    def zero_grad(self):
        self.inputs, self.h, self.new_h, self.outputs = [], [], [], []

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
        self.outputs = self.l2(self.h, self.inputs if self.residual else None)  # batch
        return self.outputs

    def backward(self, gt):
        # gt: batch of strings
        # loss
        losses = self.loss_function(self.outputs, gt)
        # l2
        l2_backward_info = [LNBackwardInfo(_i0, _i, _o, _gt, _loss) for _i0, _i, _o, _gt, _loss in zip(self.inputs, self.h, self.outputs, gt, losses)]
        _, _, _, new_h = self.l2.backward(self.task, l2_backward_info, normalize_score=self.normalize_score, is_first_layer=False, skip_good_h=self.skip_good_h)
        self.new_h = new_h
        # l1
        l1_backward_info = [LNBackwardInfo(_i0, _i, _o, _gt, _loss) for _i0, _i, _o, _gt, _loss in zip(self.inputs, self.inputs, self.h, new_h, losses)]
        _ = self.l1.backward(self.task, l1_backward_info, normalize_score=self.normalize_score, is_first_layer=True)


class Sampler(ABC):

    def __init__(self, backward_evaluate, backward_template, num_samples=4, two_step_sample_template=None):
        self.backward_evaluate = backward_evaluate
        self.backward_template = load_template(
            backward_template,
            template_directory="./templates"
        )
        self.num_samples = num_samples
        if two_step_sample_template is not None:
            self.two_step_sample_template = load_template(
                two_step_sample_template,
                template_directory="./templates"
            )
        else:
            self.two_step_sample_template = None

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class PromptSampler(Sampler):

    def parse_second_step_pi(self, input):
        # author: copilot
        pis = re.findall(r'## Instruction \d+:\n(.*?)(?=## Instruction \d+:|$)', input, re.DOTALL)
        pis = [pi.strip() for pi in pis]
        return pis

    def sample(self, task, prompt, backward_info, two_step_sample=True, **kwargs):
        """ Sample new prompts using the backward template.
            Returns a numpy array of shape (self.num_samples)
        """
        tpl_inputs = []
        for _ in range(self.num_samples):
            tpl_inputs.append(
                self.backward_template.render(
                    task=task, prompt=prompt, backward_infos=backward_info)
            )

        new_prompts = self.backward_evaluate(
            tpl_inputs,
            stop=self.backward_template.stop_tokens,
            **kwargs,
        )
        if two_step_sample is True and self.two_step_sample_template is not None:
            tpl_inputs = []
            for i in range(self.num_samples):
                tpl_inputs.append(
                    self.two_step_sample_template.render(
                        prompt=new_prompts[i], backward_infos=backward_info)
                )

            step2_sampled_prompts = self.backward_evaluate(
                tpl_inputs,
                stop=self.two_step_sample_template.stop_tokens,
                temperature=0.7,
                max_tokens=2000,
            )
            step2_sampled_prompts = ["\n".join([a, b]) for a, b in zip(tpl_inputs, step2_sampled_prompts)]  # include the first step prompt
            # parse the sampled prompts
            new_prompts = []
            for i in range(self.num_samples):
                new_prompts += self.parse_second_step_pi(step2_sampled_prompts[i])

        return np.asarray(new_prompts)


class InputSampler(Sampler):

    def parse_second_step_h(self, input):
        # author: copilot
        hs = re.findall(r'## Solution \d+:\n(.*?)(?=## Solution \d+:|$)', input, re.DOTALL)
        hs = [h.strip() for h in hs]
        return hs

    def sample(self, prompt, backward_info, **kwargs):
        """ Sample new inputs using the backward template.
            Returns a numpy array of shape (self.num_samples)
        """
        tpl_inputs = []
        for _ in range(self.num_samples):
            tpl_inputs.append(
                self.backward_template.render(
                    prompt=prompt, first_step_input=backward_info.first_step_input, input=backward_info.input, target=backward_info.target, output=backward_info.output)
            )

        sampled_inputs = self.backward_evaluate(
            tpl_inputs,
            stop=self.backward_template.stop_tokens,
            **kwargs,
        )
        if self.two_step_sample_template is not None:
            tpl_inputs = []
            for i in range(self.num_samples):
                tpl_inputs.append(
                    self.two_step_sample_template.render(
                        first_step_input=backward_info.first_step_input, input=sampled_inputs[i])
                )

            step2_sampled_inputs = self.backward_evaluate(
                tpl_inputs,
                stop=self.two_step_sample_template.stop_tokens,
                temperature=0.7,
                max_tokens=2000,
            )
            step2_sampled_inputs = ["\n".join([a, b]) for a, b in zip(tpl_inputs, step2_sampled_inputs)]  # include the first step input
            # parse the sampled inputs
            sampled_inputs = []
            for i in range(self.num_samples):
                sampled_inputs += self.parse_second_step_h(step2_sampled_inputs[i])

        return np.asarray(sampled_inputs)


class HistoryScoreCache(object):
    # this cache is used to store the most recent scores
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        # stuff is a np array of float.
        assert isinstance(stuff, np.ndarray)
        self.memory = self.memory + stuff.tolist()
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
        
    def normalize(self, stuff):
        assert isinstance(stuff, np.ndarray)
        self.push(stuff)
        mean = np.mean(np.array(self.memory))
        std = np.std(np.array(self.memory))
        output = (stuff - mean) / (std + 1e-5)
        return output

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class Scorer(ABC):

    def __init__(self, forward_evaluate, forward_template, previous_forward_template=None, eval_kwargs=None, normalzie_by_length=True):
        self.forward_evaluate = forward_evaluate
        self.forward_template = load_template(
            forward_template,
            template_directory="./templates"
        )
        self.previous_forward_template = load_template(
            previous_forward_template,
            template_directory="./templates"
        ) if previous_forward_template is not None else None
        self.eval_kwargs = eval_kwargs or {}
        self.forward_kwargs = {
            "temperature": 0,
            "max_tokens": 512,
        }
        self.normalize_by_length = normalzie_by_length
        self.y_given_pi_pool = HistoryScoreCache(capacity=1000)
        self.y_given_h_pool = HistoryScoreCache(capacity=1000)
        self.h_given_x_pool = HistoryScoreCache(capacity=1000)


class LogProbsScorer(Scorer):

    def __init__(self, forward_evaluate, forward_template, previous_forward_template=None, eval_kwargs=None, normalize_by_length=True):
        eval_kwargs = {
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "return_logprobs": True,
            "raw_logprobs": True,
        }
        super().__init__(forward_evaluate, forward_template, previous_forward_template, eval_kwargs, normalize_by_length)

    def _render_context(self, prompts, inputs, first_step_inputs, use_previous_forward_template=False):
        rendered_template = []
        for _p in prompts:
            rendered_template_per_prompt = []
            for _i, _i0 in zip(inputs, first_step_inputs):
                if use_previous_forward_template:
                    fwd_rendered = self.previous_forward_template.render(input=_i, prompt=_p)
                else:
                    fwd_rendered = self.forward_template.render(input=_i, prompt=_p, first_step_input=_i0)
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
                sum_log_probs = sum(target_log_probs)
                if self.normalize_by_length:
                    sum_log_probs = sum_log_probs / (len(target_log_probs) + 1e-5)
                output_logprobs.append(sum_log_probs)

            sum_log_probs = sum(context_log_probs)
            if self.normalize_by_length:
                sum_log_probs = sum_log_probs / (len(context_log_probs) + 1e-5)
            context_logprobs.append(sum_log_probs)

        non_empty = [o for o in output_logprobs if o != "empty"]
        if len(non_empty) == 0:
            min = 0
        else:
            min = np.min(non_empty)
        output_logprobs = [o if o != "empty" else min for o in output_logprobs]
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))  # TODO: reshape?

    def get_candidate_prompt_logprobs_contrastive(self, prompts_candidates, backward_info, normalize, **kwargs):
        score_pos = self.get_candidate_prompt_logprobs(prompts_candidates, backward_info, normalize, **kwargs)
        # now get neg scores
        inputs = [item.input for item in backward_info if item.loss > 0.0]
        wrong_outputs = [item.output for item in backward_info if item.loss > 0.0]
        first_step_inputs = [item.first_step_input for item in backward_info if item.loss > 0.0]
        if len(inputs) == 0:
            return score_pos
        num_candidates = len(prompts_candidates)
        contexts = self._render_context(prompts_candidates, inputs, first_step_inputs)  # prompts_candidates x inputs
        eval_batch = []
        for i in range(num_candidates):
            eval_batch += [f"{contexts[i][j]}\n{wrong_outputs[j]}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)
        score_neg = logprobs_results.logp_targets.reshape(
            num_candidates, len(inputs)
        ).mean(axis=-1)  # num_candidates
        if normalize:
            score_neg = self.y_given_pi_pool.normalize(score_neg)
        scores = score_pos - score_neg
        return scores

    def get_candidate_prompt_logprobs(self, prompts_candidates, backward_info, normalize, **kwargs):
        inputs = [item.input for item in backward_info]
        first_step_inputs = [item.first_step_input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        num_candidates = len(prompts_candidates)
        contexts = self._render_context(prompts_candidates, inputs, first_step_inputs)  # prompts_candidates x inputs
        eval_batch = []
        for i in range(num_candidates):
            eval_batch += [f"{contexts[i][j]}\n{gt_outputs[j]}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)
        scores = logprobs_results.logp_targets.reshape(
            num_candidates, len(inputs)
        ).mean(axis=-1)  # num_candidates
        if normalize:
            scores = self.y_given_pi_pool.normalize(scores)
        return scores

    def get_best_prompt(self, prompts_candidates, backward_info, contrastive=False, normalize=False, **kwargs):
        if contrastive:
            scores = self.get_candidate_prompt_logprobs_contrastive(prompts_candidates, backward_info, normalize, **kwargs)
        else:
            scores = self.get_candidate_prompt_logprobs(prompts_candidates, backward_info, normalize, **kwargs)
        best_indexes = scores.argmax(axis=-1)  # 1
        best_prompt = prompts_candidates[best_indexes]
        return best_prompt
    
    def get_best_k_prompt(self, prompts_candidates, backward_info, k, contrastive=False, normalize=False, **kwargs):
        # return the top-k prompts
        assert k <= len(prompts_candidates)
        if contrastive:
            scores = self.get_candidate_prompt_logprobs_contrastive(prompts_candidates, backward_info, normalize, **kwargs)
        else:
            scores = self.get_candidate_prompt_logprobs(prompts_candidates, backward_info, normalize, **kwargs)
        best_indices = scores.argsort(axis=-1)[-k:]  # k
        best_prompts = [prompts_candidates[i] for i in best_indices]
        return best_prompts
    
    def get_best_input(self, prompt, inputs, gt_output, parent_input, parent_prompt, normalize=False, phx=False, **kwargs):
        # p(y|h)
        contexts = self._render_context([prompt], inputs, [parent_input] * len(inputs))[0]  # inputs
        eval_batch = [f"{contexts[j]}\n{gt_output}" for j in range(len(inputs))]
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_y_given_h = self._get_logprobs_results(contexts, eval_results).logp_targets  # inputs
        if normalize:
            logprobs_y_given_h = self.y_given_h_pool.normalize(logprobs_y_given_h)
        logprobs_results = logprobs_y_given_h
        # p(h|x)
        if phx is True:
            assert parent_prompt is not None
            assert parent_input is not None
            if isinstance(parent_prompt, str):
                parent_prompt = [parent_prompt]  # 1 or n_nodes
            parent_input = [parent_input] * len(inputs)  # inputs
            contexts = self._render_context(parent_prompt, parent_input, parent_input, use_previous_forward_template=True)  # parent_prompt x parent_input
            eval_batch = []
            for i in range(len(parent_prompt)):
                eval_batch += [f"{contexts[i][j]}\n{inputs[j]}" for j in range(len(inputs))]
            eval_results = self._forward_unique_evals(eval_batch)
            logprobs_h_given_x = self._get_logprobs_results(contexts, eval_results).logp_targets  # parent_prompt x inputs
            logprobs_h_given_x = logprobs_h_given_x.reshape(len(parent_prompt), len(inputs)).mean(axis=0)  # inputs
            if normalize:
                logprobs_h_given_x = self.h_given_x_pool.normalize(logprobs_h_given_x)
            logprobs_results = logprobs_results + logprobs_h_given_x
        best_indexes = logprobs_results.argmax(axis=-1)  # 1
        best_input = inputs[best_indexes]
        return best_input

    def get_candidate_prompt_logprobs4WideSummary_contrastive(self, prompts_candidates, backward_info, aggregation_forward_template, normalize, **kwargs):
        score_pos = self.get_candidate_prompt_logprobs4WideSummary(prompts_candidates, backward_info, aggregation_forward_template, normalize, **kwargs)
        # now get neg scores
        inputs = [item.input for item in backward_info if item.loss > 0.0]
        wrong_outputs = [item.output for item in backward_info if item.loss > 0.0]
        if len(inputs) == 0:
            return score_pos
        # prompts_candidates: num_nodes x num_candidates per node
        # inputs: num_inputs
        # wrong_outputs: num_inputs
        num_nodes = len(prompts_candidates)
        num_candidates_per_node = len(prompts_candidates[0])
        num_inputs = len(inputs)  # also the number of gt_outputs
        merged_context = []
        for i in range(num_nodes):
            _first_step_context = self._render_context([prompts_candidates[i][j] for j in range(num_candidates_per_node)], inputs)  # num_candidates_per_node x inputs
            for j in range(num_candidates_per_node):
                merged_context += _first_step_context[j]  # inputs
        # merged_context: num_nodes*num_candidates*inputs
        merged_h = self._forward_unique_evals(merged_context, forward=True)
        # split the merged_h into num_nodes x num_nodes*num_candidates
        h_list = []
        for i in range(num_nodes):
            h_list.append(merged_h[i*num_candidates_per_node*num_inputs:(i+1)*num_candidates_per_node*num_inputs])
        # h_list: num_nodes x num_candidates_per_node*inputs
        # now aggregate the h_list
        eval_batch = []
        wrong_outputs_expand = wrong_outputs * num_candidates_per_node
        contexts = []
        for i in range(len(h_list[0])):  # num_candidates_per_node*inputs
            _inputs = [h_list[j][i] for j in range(num_nodes)]
            _inputs = aggregation_forward_template.render(inputs=_inputs)
            contexts.append(_inputs)
            eval_batch.append(f"{_inputs}\n{wrong_outputs_expand[i]}")
        eval_results = self._forward_unique_evals(eval_batch)
        logprobs_results = self._get_logprobs_results(contexts, eval_results)  # num_candidates*inputs
        score_neg = logprobs_results.logp_targets.reshape(
            num_candidates_per_node, num_inputs
        ).mean(axis=-1)  # num_candidates
        if normalize:
            score_neg = self.y_given_pi_pool.normalize(score_neg)
        scores = score_pos - score_neg
        return scores

    def get_candidate_prompt_logprobs4WideSummary(self, prompts_candidates, backward_info, aggregation_forward_template, normalize, **kwargs):
        inputs = [item.input for item in backward_info]
        gt_outputs = [item.target for item in backward_info]
        # prompts_candidates: num_nodes x num_candidates per node
        # inputs: num_inputs
        # gt_outputs: num_inputs
        num_nodes = len(prompts_candidates)
        num_candidates_per_node = len(prompts_candidates[0])
        num_inputs = len(inputs)  # also the number of gt_outputs

        merged_context = []
        for i in range(num_nodes):
            _first_step_context = self._render_context([prompts_candidates[i][j] for j in range(num_candidates_per_node)], inputs)  # num_candidates_per_node x inputs
            for j in range(num_candidates_per_node):
                merged_context += _first_step_context[j]  # inputs
        # merged_context: num_nodes*num_candidates*inputs
        merged_h = self._forward_unique_evals(merged_context, forward=True)
        # split the merged_h into num_nodes x num_nodes*num_candidates
        h_list = []
        for i in range(num_nodes):
            h_list.append(merged_h[i*num_candidates_per_node*num_inputs:(i+1)*num_candidates_per_node*num_inputs])
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

        score_pos = logprobs_results.logp_targets.reshape(
            num_candidates_per_node, num_inputs
        ).mean(axis=-1)  # num_candidates
        if normalize:
            score_pos = self.y_given_pi_pool.normalize(score_pos)
        return score_pos

    def get_best_prompt4WideSummary(self, prompts_candidates, backward_info, aggregation_forward_template, contrastive=False, normalize=False, **kwargs):
        if contrastive:
            scores = self.get_candidate_prompt_logprobs4WideSummary_contrastive(prompts_candidates, backward_info, aggregation_forward_template, normalize, **kwargs)
        else:
            scores = self.get_candidate_prompt_logprobs4WideSummary(prompts_candidates, backward_info, aggregation_forward_template, normalize, **kwargs)
        best_indexes = scores.argmax(axis=-1)  # 1
        best_prompt = [item[best_indexes] for item in prompts_candidates]  # num_nodes
        return best_prompt
