from argparse import ArgumentParser
import datetime
import logging
import os
import tqdm
import re
import numpy as np
from collections import Counter
from termcolor import colored

from dln.dataset import Dataset, init_dataset

# from dln.loss import LossRegistry
from dln.operator import LLMRegistry

from dln.vi.model import log_message
from layers import DLN_2


def gsm8k_loss(y_hat, y):
    # y_hat: a list of strings
    # y: a list of strings
    assert len(y_hat) == len(y)
    loss = []
    for i in range(len(y_hat)):
        y_numbers = re.findall(r'\b\d+\b', y[i])
        y_hat_numbers = re.findall(r'\b\d+\b', y_hat[i])
        if len(set(y_numbers) & set(y_hat_numbers)) > 0:
            loss.append(0.0)
        else:
            loss.append(1.0)
    return loss


def subj_loss(y_hat, y):
    # y_hat: a list of strings
    # y: a list of strings
    assert len(y_hat) == len(y)
    loss = []
    for i in range(len(y_hat)):
        _y_hat, _y = y_hat[i].lower().strip(), y[i].lower().strip()
        if len(_y_hat) > 0 and _y_hat == _y:
            loss.append(0.0)
        else:
            loss.append(1.0)
    return loss


def validate(model, dataset: Dataset, iteration):

    log_message("===================================")
    log_message(colored("VALIDATING... ITER %s" % str(iteration), "red"))
    log_message("Current L1 weights:", model.l1.prompt, "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))
    log_message("Current L2 weights:", model.l2.prompt, "\n-- This layer is " + ("trainable" if model.l2.trainable else "fixed"))

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
        if dataset.dataset_name == "gsm8k":
            losses = gsm8k_loss(y_hat, y)
        elif dataset.dataset_name == "subj":
            losses = subj_loss(y_hat, y)
        else:
            raise NotImplementedError

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


def train(model, dataset: Dataset, batch_size, iters):

    dev_acc = []
    _acc = validate(model, dataset, 0)
    dev_acc.append(_acc)
    for iter_num in range(iters):
        x, y, _ = dataset.get_batch("train", batch_size, random_sample=True)
        y_hat = model.forward(x)
        h, input, = model.h, model.inputs
        model.backward(y)
        log_message("===================================")
        log_message(colored("------- L1", "red"))
        log_message(colored(model.l1.prompt, "red"))
        log_message(colored("------- L2", "red"))
        log_message(colored(model.l2.prompt, "red"))
        for i, (a, b, c, d) in enumerate(zip(input, h, y_hat, y)):
            log_message("-------------------------------" + str(i))
            log_message(f"--------------\nx: {a}\nh: {b}\ny_hat: {c}\ny: {d}\n")
        
        import pdb; pdb.set_trace()
        _acc = validate(model, dataset, iter_num + 1)
        dev_acc.append(_acc)
        log_message("===================================")
        log_message(colored("DEV ACC", "blue"))
        log_message(colored(str(dev_acc), "blue"))


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
    if args.dataset == "subj":
        task_info_str = "Read the following sentence, then choose whether it is subjective or objective."
    elif args.dataset == "gsm8k":
        task_info_str = "Solve the math world problem."
    else:
        raise NotImplementedError
    model = DLN_2(task_info_str, fwd_model, bwd_model)

    train(
        model=model,
        dataset=dataset,
        # loss_fn=loss_fn,
        batch_size=args.batch_size,
        iters=args.iters
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--fwd_model", type=str)
    parser.add_argument("--bwd_model", type=str)
    parser.add_argument("--output_scoring_function", type=str)
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max_train_size", type=int, default=200)
    parser.add_argument("--max_dev_size", type=int, default=200)
    parser.add_argument("--max_test_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./log", help="log directory")
    args = parser.parse_args()
    train_dln(args)


if __name__ == "__main__":
    main()

# python main.py --config llm_config.yaml --fwd_model gpt-3-fwd --bwd_model gpt-3-bwd --dataset subj --output_scoring_function accuracy --out_dir log/debug
