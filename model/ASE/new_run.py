import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from ASEDataset import TestFileDataset, TrainFileDataset, TgTestFileDataset, TgTrainFileDataset

from Trec_Metrics import Metrics
from ASE_model import ASEModel
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartTokenizer, BartForConditionalGeneration, BertTokenizer
from utils import load_geohash


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="fnlp/bart-base-chinese",
                    type=str,
                    help="Directory of pre-trained model.")
parser.add_argument("--config_name",
                    default="fnlp/bart-base-chinese",
                    type=str,
                    help="Directory of the config of the pre-trained model.")
parser.add_argument("--tokenizer_name",
                    default="bert-base-chinese",
                    type=str,
                    help="Directory of the tokenizer of the pre-trained model.")
parser.add_argument("--output_dir",
                    default="./output/model/",
                    type=str,
                    help="Directory of the output checkpoints.")
parser.add_argument("--result_dir",
                    default="./output/",
                    type=str,
                    help="Directory of the output scores.")
parser.add_argument("--dataset",
                    default="poi",
                    type=str,
                    help="Which data")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="")
parser.add_argument("--num_train_epochs",
                    default=5,
                    type=int,
                    help="")
parser.add_argument("--max_steps",
                    default=-1,
                    type=int,
                    help="Max steps.")
parser.add_argument("--warmup_steps",
                    default=0,
                    type=int,
                    help="Warm steps.")
parser.add_argument("--logging_steps",
                    default=500,
                    type=int,
                    help="Steps for logging.")
parser.add_argument("--eval_steps",
                    default=-1,
                    type=int,
                    help="Evaluating steps")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--seed",
                    default=0,
                    type=int,
                    help="Random seed for reproducibility.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--score_file_pre_path",
                    default="score_file.pre.txt",
                    type=str,
                    help="The path to save results.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save results.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument("--do_train",
                    default='True',
                    type=str2bool,
                    help="")
parser.add_argument("--do_eval",
                    default='True',
                    type=str2bool,
                    help="")
args = parser.parse_args()

args.log_path += ASEModel.__name__ + '.' + args.dataset + ".log"
logger = open(args.log_path, "a")
args.ckp_path = args.save_path + ASEModel.__name__ + "-" +  args.dataset
args.save_path += ASEModel.__name__ + "." +  args.dataset
result_path = "./output/" + args.dataset + "/"
score_file_prefix = result_path + ASEModel.__name__ + "." + args.dataset
args.score_file_path = score_file_prefix + "." +  args.score_file_path
args.score_file_pre_path = score_file_prefix + "." +  args.score_file_pre_path
print(args)

# Set seed for reproducibility.
def set_seed(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

dict_geohash = load_geohash('./data/poi/1_geohash_code.csv')
# Add special tokens.
tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
geo_special_tokens_dict = ['[' + gcd + ']' for gcd in dict_geohash]
tokenizer.add_tokens(geo_special_tokens_dict)
# Add special tokens.
tokenizer.add_tokens("[eos]")
tokenizer.add_tokens("[empty_d]")
tokenizer.add_tokens("[empty_q]")
tokenizer.add_tokens("[rank]")
tokenizer.add_tokens("[genfq]")
tokenizer.add_tokens("[gencd]")
tokenizer.add_tokens("[gensq]")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data.
train_data = "./data/" + args.dataset + "/train.txt"
test_data = "./data/" + args.dataset + "/test.txt"

def predict(model, X_test):
    model.eval()
    test_dataset = TestFileDataset(X_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    y_pred = []
    y_label = []
    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data,  is_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["ranking_labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
            break
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    return y_pred, y_label


def train_step(model, batches):
    with torch.no_grad():
        for batch in batches:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

    gen_loss, rank_loss = model.forward(batches)

    return gen_loss, rank_loss


def evaluate(model, X_test, best_result, epoch, step, is_test=False):
    if args.dataset == "poi":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=51)

    elif args.dataset == "tiangong":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=10)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')

    result = metrics.evaluate_all_metrics()

    if not is_test and result[0] + result[1] + result[2] + result[3] + result[4] + result[5] > best_result[0] + \
            best_result[1] + best_result[2] + best_result[3] + best_result[4] + best_result[5]:
        best_result = result
        print(
            "Epoch: %d, Step: %d, Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (
            epoch, step, best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],
            best_result[5]))
        logger.write(
            "Epoch: %d, Step: %d, Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (
            epoch, step, best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],
            best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    if is_test:
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (
        result[0], result[1], result[2], result[3], result[4], result[5]))
    return best_result

def fit(model, X_train, X_test):
    train_dataset = TrainFileDataset(X_train, tokenizer, args.dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs
    if args.eval_steps < 0:
        args.eval_steps = len(train_dataloader) // 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    print("Num Examples = ", len(train_dataset))
    print("Num Epochs = ", args.num_train_epochs)
    print("Total Optimization Steps = ", t_total)

    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    global_step = 0
    for epoch in range(args.num_train_epochs):
        print("\nEpoch ", epoch + 1, "/", args.num_train_epochs)
        model.train()
        total_loss, total_gen_loss, total_rank_loss = 0.0, 0.0, 0.0
        tmp_loss, tmp_gen_loss, tmp_rank_loss = 0.0, 0.0, 0.0
        for step, (batches) in tqdm(enumerate(train_dataloader)):
            gen_loss, rank_loss = train_step(model, batches)
            gen_loss = gen_loss.mean()
            rank_loss = rank_loss.mean()

            loss = gen_loss + rank_loss
            loss.backward()

            total_loss = total_loss + loss.item()
            total_gen_loss = total_gen_loss + gen_loss.item()
            total_rank_loss = total_rank_loss + rank_loss.item()
            optimizer.step()
            scheduler.step()

            model.zero_grad()
            global_step += 1
            if step > 0 and step % args.logging_steps == 0:
                print(
                    "Step = {:d}\tLR = {:.6f}\tTotal Loss = {:.6f}\tGen Loss = {:.6f}\tRank Loss = {:.6f}".format(step,
                                                                                                                  scheduler.get_last_lr()[
                                                                                                                      0],
                                                                                                                  (
                                                                                                                              total_loss - tmp_loss) / args.logging_steps,
                                                                                                                  (
                                                                                                                              total_gen_loss - tmp_gen_loss) / args.logging_steps,
                                                                                                                  (
                                                                                                                              total_rank_loss - tmp_rank_loss) / args.logging_steps))
                tmp_loss = total_loss
                tmp_gen_loss = total_gen_loss
                tmp_rank_loss = total_rank_loss
            if args.do_eval and step > 0 and args.eval_steps > 0 and step % args.eval_steps == 0:
                print(args.eval_steps)
                print("Step = {:d}\tStart Evaluation".format(step, scheduler.get_lr()[0]))
                cnt = len(train_dataset) // args.batch_size + 1
                tqdm.write("Average loss:{:.6f} ".format(total_loss / cnt))
                best_result = evaluate(model, X_test, best_result, epoch, step)
                model.train()
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        print("Epoch = {:d}\tLoss = {:.6f}".format(epoch + 1, total_loss / len(train_dataloader)))
        if args.max_steps > 0 and global_step > args.max_steps:
            break

# Train.
def train_model():
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart.resize_token_embeddings(len(tokenizer))
    model = ASEModel(bart, tokenizer)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("* number of parameters: %d" % n_params)
    model = model.to(device)
    fit(model, train_data, train_data)

def test_model():
    config = BartConfig.from_pretrained(args.config_name)
    bart_model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart_model.resize_token_embeddings(len(tokenizer))
    model = ASEModel(bart_model, tokenizer)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    evaluate(model, test_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0, is_test=True)


if __name__ == '__main__':
    set_seed(args.seed)
    if args.do_train:
        train_model()
    elif args.do_eval:
        test_model()