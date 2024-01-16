import argparse
from argparse import ArgumentParser
import datetime
import logging
import os
import tqdm
import re
import yaml
import numpy as np
from collections import Counter
from termcolor import colored

from dln.dataset import Dataset, init_dataset
from dln.loss import NumberPresenceLoss, ExactMatchLoss
from dln.postprocessing import postprocess_prediction

# from dln.loss import LossRegistry
from dln.operator import LLMRegistry

from dln.vi.model import log_message
from layers import DLN_2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate(model, loss_function, dataset: Dataset, iteration):

    log_message("===================================")
    log_message(colored("VALIDATING... ITER %s" % str(iteration), "red"))
    log_message("Current L1 weights:\n", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))
    log_message("Current L2 weights:\n", model.l2.prompt_print(), "\n-- This layer is " + ("trainable" if model.l2.trainable else "fixed"))

    acc = 0.0
    tot = 0.0
    pbar = tqdm.tqdm(
        total=dataset.dev_size,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Eval",
    )
    dataset.reset_pointer("dev")

    for batch in dataset.iterate("dev", batch_size=20):
        x, y, _ = batch
        y_hat = model.forward(x)
        losses = loss_function(y_hat, y)

        acc += len(y) - np.sum(losses)
        tot += len(y)

        pbar.update(len(y))
        pbar.set_postfix_str(f"{acc / tot:.1%}")
    
    dev_acc = acc / tot
    if iteration == 0:
        log_message(colored("INIT DEV ACC: {}".format(dev_acc), "red"))
    else:
        log_message(colored("DEV ACC: {}".format(dev_acc), "red"))
    return dev_acc


def test(model, loss_function, dataset: Dataset):

    log_message("===================================")
    log_message(colored("TESTING... ", "red"))
    log_message("Current L1 weights:\n", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))
    log_message("Current L2 weights:\n", model.l2.prompt_print(), "\n-- This layer is " + ("trainable" if model.l2.trainable else "fixed"))

    acc = 0.0
    tot = 0.0
    pbar = tqdm.tqdm(
        total=dataset.dev_size,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Eval",
    )
    dataset.reset_pointer("test")

    for batch in dataset.iterate("test", batch_size=20):
        x, y, _ = batch
        y_hat = model.forward(x)
        losses = loss_function(y_hat, y)

        acc += len(y) - np.sum(losses)
        tot += len(y)

        pbar.update(len(y))
        pbar.set_postfix_str(f"{acc / tot:.1%}")
    
    test_acc = acc / tot
    log_message(colored("TEST ACC: {}".format(test_acc), "red"))
    return test_acc


def train(model, loss_function, dataset: Dataset, batch_size, iters, patience):

    dev_acc = []
    _acc = validate(model, loss_function, dataset, 0)
    dev_acc.append(_acc)
    best_model = model.save_model()
    log_message(colored("Saving model...", "red"))
    best_acc = _acc
    patience_counter = 0

    for iter_num in range(iters):
        x, y, _ = dataset.get_batch("train", batch_size, random_sample=True)
        y_hat = model.forward(x)
        h, input, = model.h, model.inputs
        model.backward(y)
        new_h = model.new_h
        log_message("===================================")
        log_message(colored("------- L1", "red"))
        log_message(colored(model.l1.prompt_print(), "red"))
        log_message(colored("------- L2", "red"))
        log_message(colored(model.l2.prompt_print(), "red"))
        for i, (a, b, c, d, e) in enumerate(zip(input, h, new_h, y_hat, y)):
            if b == c:
                c = "--"
            log_message("-------------------------------" + str(i))
            log_message(f"--------------\n**x:**\n{a}\n\n**h:**\n{b}\n\n**new_h:**\n{c}\n\n**y_hat:**\n{d}\n\n**y:**\n{e}\n\n")

        model.zero_grad()
        _acc = validate(model, loss_function, dataset, iter_num + 1)
        dev_acc.append(_acc)
        log_message("===================================")
        log_message(colored("DEV ACC", "blue"))
        log_message(colored(str(dev_acc), "blue"))
        if patience > 0:  # 0 means disabled
            if _acc < best_acc:
                patience_counter += 1
                if patience_counter >= patience:
                    log_message(colored("Loading best model...", "red"))
                    model.load_model(best_model)
                    patience_counter = 0
            else:
                best_model = model.save_model()
                log_message(colored("Saving model...", "red"))
                best_acc = _acc
                patience_counter = 0

    model.load_model(best_model)
    log_message("===================================")
    log_message(colored("BEST DEV ACC: %s" % str(best_acc), "red"))
    test_acc = test(model, loss_function, dataset)
    # test_acc = 0.0
    log_message(colored("TEST ACC: %s" % str(test_acc), "red"))
    log_message(colored("BEST MODEL:", "red"))
    log_message("L1 weights:\n", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))
    log_message("L2 weights:\n", model.l2.prompt_print(), "\n-- This layer is " + ("trainable" if model.l2.trainable else "fixed"))


def train_dln(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    output_log_dir = os.path.join(out_dir, "output.log")
    logging.basicConfig(
        filename=output_log_dir,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_message(str(args))
    log_message(f"Logging to... {output_log_dir}")

    dataset = init_dataset(
        dataset_id=args.dataset,
        seed=args.seed,
        data_dir=args.data_dir,
        max_train_size=args.max_train_size,
        max_dev_size=args.max_dev_size,
        max_test_size=args.max_test_size,
    )

    llm_registry = LLMRegistry.from_yaml(args.config)
    fwd_model = llm_registry[args.fwd_model]
    bwd_model = llm_registry[args.bwd_model]

    # loss_fn = LossRegistry.instantiate(args.loss_function)
    if args.dataset == "gsm8k":
        task_info_str = "Solve the math word problem."
        loss_fn = NumberPresenceLoss()
    else:
        with open("../../dln/dataset_info.yaml") as reader:
            task_info_dict = yaml.safe_load(reader)
            if args.dataset in task_info_dict:
                task_info_str = task_info_dict[args.dataset]["instruction"]
            else:
                raise ValueError(f"Dataset {args.dataset} not found in dln/dataset_info.yaml")
            loss_fn = ExactMatchLoss(postproc=postprocess_prediction)

    model = DLN_2(task_info_str, fwd_model, bwd_model, loss_fn, num_samples=args.num_samples,
                   prompt_backward_template=args.prompt_backward_template, input_backward_template=args.input_backward_template,
                   first_layer_contrastive=args.first_layer_contrastive, score_input_phx=args.score_input_phx,
                   normalize_score=args.normalize_score, skip_good_h=args.skip_good_h, normalize_by_length=args.normalize_by_length,
                   two_step_h_sample=args.two_step_h_sample, two_step_pi_sample=args.two_step_pi_sample, residual=args.residual)

    train(
        model=model,
        loss_function=loss_fn,
        dataset=dataset,
        batch_size=args.batch_size,
        iters=args.iters,
        patience=args.patience
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--fwd_model", type=str)
    parser.add_argument("--bwd_model", type=str)
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--prompt_backward_template", type=str, default="ln_prompt_backward:1.0")
    parser.add_argument("--input_backward_template", type=str, default="ln_input_backward:1.0")
    parser.add_argument("--max_train_size", type=int, default=400)
    parser.add_argument("--max_dev_size", type=int, default=200)
    parser.add_argument("--max_test_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--first_layer_contrastive", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--score_input_phx", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--normalize_score", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--skip_good_h", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--normalize_by_length", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--two_step_h_sample", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--two_step_pi_sample", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--residual", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--out_dir", type=str, default="./log", help="log directory")
    args = parser.parse_args()
    train_dln(args)


if __name__ == "__main__":
    main()

# python dln2.py --config llm_config.yaml --fwd_model gpt-3-fwd --bwd_model gpt-3-bwd --dataset gsm8k --out_dir log/debug --max_train_size 400 --batch_size 20 --iters 50 --patience 2 --num_samples 10
