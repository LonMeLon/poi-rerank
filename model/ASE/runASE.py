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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#BartChinese
#BARTBase
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="./pretrained-model/BARTBase/",
                    type=str,
                    help="Directory of pre-trained model.")
parser.add_argument("--config_name",
                    default="./pretrained-model/BARTBase/",
                    type=str,
                    help="Directory of the config of the pre-trained model.")
parser.add_argument("--tokenizer_name",
                    default="./pretrained-model/BARTBase/",
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
                    default="aol",
                    type=str,
                    help="Which data set aol/tiangong.")
parser.add_argument("--do_train",
                    default='True',
                    type=str2bool,
                    help="")
parser.add_argument("--do_eval",
                    default='True',
                    type=str2bool,
                    help="")
parser.add_argument("--per_gpu_train_batch_size",
                    default=32,
                    type=int,
                    help="")
parser.add_argument("--per_gpu_eval_batch_size",
                    default=64,
                    type=int,
                    help="")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="")
parser.add_argument("--max_grad_norm",
                    default=1.0,
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
args = parser.parse_args()
args.batch_size = args.per_gpu_train_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_train_batch_size
args.test_batch_size = args.per_gpu_eval_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_eval_batch_size

# Load tokenizers. 
# Note that the Chinese BART provided in Hugging face (https://huggingface.co/fnlp/bart-base-chinese) uses BertTokenizer.
if args.dataset == 'aol':
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
elif args.dataset == 'tiangong':
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

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
args.log_path += ASEModel.__name__ + '.' + args.dataset + ".log"
logger = open(args.log_path, "a")
args.ckp_path = args.save_path + ASEModel.__name__ + "-" +  args.dataset
args.save_path += ASEModel.__name__ + "." +  args.dataset
result_path = "./output/" + args.dataset + "/"
score_file_prefix = result_path + ASEModel.__name__ + "." + args.dataset
args.score_file_path = score_file_prefix + "." +  args.score_file_path
args.score_file_pre_path = score_file_prefix + "." +  args.score_file_pre_path
print(args)

# Data.
train_data = "./data/" + args.dataset + "/train.txt"
dev_data = "./data/" + args.dataset + "/dev.txt"
test_data = "./data/" + args.dataset + "/test.txt"

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

# Train.
def train_model():
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart.resize_token_embeddings(len(tokenizer))
    model = ASEModel(bart, tokenizer)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("* number of parameters: %d" % n_params)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, dev_data)

def fit(model, X_train, X_test):
    train_dataset = TrainFileDataset(X_train, tokenizer, args.dataset)
    if args.dataset == 'tiangong':
        train_dataset = TgTrainFileDataset(X_train, tokenizer, args.dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
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
    print("Batch Size per GPU = ", args.per_gpu_train_batch_size)
    print("Total Train Batch Size = ", args.batch_size)
    print("Total Optimization Steps = ", t_total)

    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    global_step = 0
    for epoch in range(args.num_train_epochs):
        print("\nEpoch ", epoch + 1, "/", args.num_train_epochs)
        model.train()
        total_loss, total_gen_loss, total_rank_loss = 0.0, 0.0, 0.0
        tmp_loss, tmp_gen_loss, tmp_rank_loss = 0.0, 0.0, 0.0
        epoch_iterator = tqdm(train_dataloader, ncols=120, position=0, leave=True)
        for step, (batches) in enumerate(epoch_iterator):
            gen_loss, rank_loss = train_step(model, batches)
            gen_loss = gen_loss.mean()
            rank_loss = rank_loss.mean()

            loss = gen_loss + rank_loss
            loss.backward()

            total_loss = total_loss + loss.item()
            total_gen_loss = total_gen_loss + gen_loss.item()
            total_rank_loss = total_rank_loss + rank_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            model.zero_grad()
            global_step += 1
            if step > 0 and step % args.logging_steps == 0:
                print("Step = {:d}\tLR = {:.6f}\tTotal Loss = {:.6f}\tGen Loss = {:.6f}\tRank Loss = {:.6f}".format(step, scheduler.get_last_lr()[0], (total_loss - tmp_loss) / args.logging_steps, (total_gen_loss - tmp_gen_loss) / args.logging_steps, (total_rank_loss - tmp_rank_loss) / args.logging_steps))
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

def train_step(model, batches):
    with torch.no_grad():
        for batch in batches:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
    
    gen_loss, rank_loss = model.forward(batches)

    return gen_loss, rank_loss

def evaluate(model, X_test, best_result, epoch, step, is_test=False):
    if args.dataset == "aol":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=50)
    elif args.dataset == "tiangong":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=10)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')
            
    result = metrics.evaluate_all_metrics()

    if not is_test and result[0] + result[1] + result[2] + result[3] + result[4] + result[5] > best_result[0] + best_result[1] + best_result[2] + best_result[3] + best_result[4] + best_result[5]:
        best_result = result
        print("Epoch: %d, Step: %d, Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (epoch, step, best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.write("Epoch: %d, Step: %d, Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (epoch, step, best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    if is_test:
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (result[0], result[1], result[2], result[3], result[4], result[5]))
    return best_result


def predict(model, X_test):
    model.eval()
    test_dataset = TestFileDataset(X_test, tokenizer)
    if args.dataset == "tiangong":
        test_dataset = TgTestFileDataset(X_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data,  is_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["ranking_labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    return y_pred, y_label

def test_model():
    config = BartConfig.from_pretrained(args.config_name)
    bart_model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart_model.resize_token_embeddings(len(tokenizer))
    model = ASEModel(bart_model, tokenizer)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate(model, test_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0, is_test=True)

if __name__ == '__main__':
    set_seed(args.seed)
    if args.do_train:
        train_model()
    elif args.do_eval:
        test_model()
