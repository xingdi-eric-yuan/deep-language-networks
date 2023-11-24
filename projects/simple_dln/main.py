from argparse import ArgumentParser
import datetime
import logging
import os

from dln.dataset import Dataset, init_dataset

# from dln.loss import LossRegistry
from dln.operator import LLMRegistry

from dln.vi.model import log_message
from layers import DLN_2


# try:
#     from wandb.integration.openai import autolog
#     autolog({"project":"dwln"})
# except ImportError:
#     pass


def train(model, dataset: Dataset, batch_size, iters):
    for _ in range(iters):
        x, y, _ = dataset.get_batch("train", batch_size, random_sample=True)
        y_hat = model.forward(x)
        h, input, = model.h, model.input
        model.backward(y)
        print("===================================")
        print("------- L1")
        print(model.l1.prompt)
        print("------- L2")
        print(model.l2.prompt)
        for i, (a, b, c, d) in enumerate(zip(input, h, y_hat, y)):
            print("-------------------------------" + str(i))
            print(f"--------------\nx: {a}\nh: {b}\ny_hat: {c}\ny: {d}\n")


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
    model = DLN_2("Solve the math world problem", fwd_model, bwd_model)

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

# python main.py --config llm_config.yaml --fwd_model gpt-3-fwd --bwd_model gpt-3-bwd --dataset gsm8k --output_scoring_function accuracy --out_dir log/debug
