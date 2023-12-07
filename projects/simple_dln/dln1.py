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
from dln.loss import NumberPresenceLoss

# from dln.loss import LossRegistry
from dln.operator import LLMRegistry

from dln.vi.model import log_message
from layers import DLN_1


def validate(model, dataset: Dataset, iteration):

    log_message("===================================")
    log_message(colored("VALIDATING... ITER %s" % str(iteration), "red"))
    log_message("Current L1 weights:", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))

    acc = 0.0
    tot = 0.0
    pbar = tqdm.tqdm(
        total=dataset.dev_size,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Eval",
    )
    dataset.reset_pointer("dev")
    loss_function = NumberPresenceLoss()

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


def test(model, dataset: Dataset):

    log_message("===================================")
    log_message(colored("TESTING... ", "red"))
    log_message("Current L1 weights:", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))

    acc = 0.0
    tot = 0.0
    pbar = tqdm.tqdm(
        total=dataset.dev_size,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Eval",
    )
    dataset.reset_pointer("test")
    loss_function = NumberPresenceLoss()

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


def train(model, dataset: Dataset, batch_size, iters, patience):

    dev_acc = []
    _acc = validate(model, dataset, 0)
    dev_acc.append(_acc)
    best_model = model.save_model()
    log_message(colored("Saving model...", "red"))
    best_acc = _acc
    patience_counter = 0

    for iter_num in range(iters):
        x, y, _ = dataset.get_batch("train", batch_size, random_sample=True)
        y_hat = model.forward(x)
        input = model.inputs
        model.backward(y)
        log_message("===================================")
        log_message(colored("------- L1", "red"))
        log_message(colored(model.l1.prompt_print(), "red"))
        for i, (a, b, c) in enumerate(zip(input, y_hat, y)):
            log_message("-------------------------------" + str(i))
            log_message(f"--------------\nx: {a}\ny_hat: {b}\ny: {c}\n")

        model.zero_grad()
        _acc = validate(model, dataset, iter_num + 1)
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
    test_acc = test(model, dataset)
    log_message(colored("TEST ACC: %s" % str(test_acc), "red"))
    log_message(colored("BEST MODEL:", "red"))
    log_message("L1 weights:", model.l1.prompt_print(), "\n-- This layer is " + ("trainable" if model.l1.trainable else "fixed"))


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
    model = DLN_1(task_info_str, fwd_model, bwd_model, num_samples=args.num_samples)

    train(
        model=model,
        dataset=dataset,
        # loss_fn=loss_fn,
        batch_size=args.batch_size,
        iters=args.iters,
        patience=args.patience
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
    parser.add_argument("--max_test_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./log", help="log directory")
    args = parser.parse_args()
    train_dln(args)


if __name__ == "__main__":
    main()

# python dln1.py --config llm_config.yaml --fwd_model gpt-3-fwd --bwd_model gpt-3-bwd --dataset gsm8k --output_scoring_function accuracy --out_dir log/debug --max_train_size 400 --batch_size 20 --iters 50 --patience 2 --num_samples 10
