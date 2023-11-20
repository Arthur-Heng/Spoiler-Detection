import torch
import numpy as np
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from main_model import MVSD
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as auc
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import init_weights
import random
import argparse

parser = argparse.ArgumentParser(description='hyperparameters')
parser.add_argument('--EPOCH', type=int, default = 70)
parser.add_argument('--LEARNING_RATE', type=float, default = 1e-3)
parser.add_argument('--DROPOUT', type=float, default = 0.3)
parser.add_argument('--WEIGHT_DECAY',type=float, default = 1e-5)
parser.add_argument('--BATCH_SIZE', type=int, default = 1024)
parser.add_argument('--IN_DIM', type=int, default = 768)
parser.add_argument('--HIDDEN_DIM', type=int, default = 128)
parser.add_argument('--device',default = 0)
parser.add_argument('--N_SD',type=int, default = 2, help='number of MVSD layers (default: 2)')
parser.add_argument('--N_LAYER',type=int, default = 1)
parser.add_argument('--c_c',type=int, default = 1)
parser.add_argument('--int_type',type=int, default = 2, help='type of interaction (default: 2)')
parser.add_argument('--conv',type=int, default = 2)
parser.add_argument('--patience',type=int, default = 5)
parser.add_argument('--ps', type=str, default = "None", help='extra information adding to log (default: None)')
args = parser.parse_args()
print(args)
device = torch.device('cuda:'+str(args.device))

# Constant for the Kaggle dataset (i.e. index for the nodes)
year0, year_num = 0, 98
rating0, rating_num = 98, 7
cast0, cast_num =  105, 7865
genre0, genre_num = 7970, 21
movie0, movie_num = 7991, 1572
rating2, rating2_num = 9563, 7
review0, review_num = 9570, 573913
user0, user_num = 583483,259705
year2, year2_num = 843188, 25
graph1_node = 9562
graph2_node = 583482
graph3_node = 843212
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     #torch.backends.cudnn.deterministic = True
#setup_seed(3412)

year1, year2, rating1, rating2 = nn.Parameter(torch.randn(year_num,768)),nn.Parameter(torch.randn(year2_num,768)),nn.Parameter(torch.randn(rating_num,768)),nn.Parameter(torch.randn(rating2_num,768))
year_meta, rating_meta = nn.Parameter(torch.randn(year2_num,5)),nn.Parameter(torch.randn(rating2_num,5))

def load_data(path="../feature/", *argv):
    print("loading features...")
    text_features = torch.load(path + "text.pt", map_location="cpu")
    meta_features = torch.load(path + "meta.pt", map_location="cpu")
    struct_features  = torch.load(path + "structural.pt", map_location="cpu")

    print("feature scaling...")
    meta_features[:movie_num] = torch.tensor(scale(meta_features[:movie_num]))
    meta_features[movie_num:movie_num + review_num] = torch.tensor(scale(meta_features[movie_num:movie_num + review_num]))
    meta_features[movie_num + review_num:] =  torch.tensor(scale(meta_features[movie_num + review_num:]))
    meta_features = torch.cat((meta_features[:movie_num],rating_meta,meta_features[movie_num:],year_meta),0)
    meta_features = torch.cat((torch.zeros(movie0,5),meta_features),0)

    text_features = torch.cat((year1,rating1,text_features[:graph1_node-year_num-rating_num+1],rating2,text_features[graph1_node-year_num-rating_num+1:],year2),0)

    struct_features = torch.cat((struct_features, torch.zeros(graph3_node - graph1_node, 768)),0)

    #concatenate the three views of feature first
    features = torch.cat((text_features, meta_features, struct_features),1)
    print("loading edges & label...")
    edge_index = torch.load(path + "edge_index.pt", map_location="cpu")
    edge_type = torch.load(path + "edge_type.pt", map_location="cpu")
    label = torch.load(path + "label.pt", map_location = "cpu").to(torch.int64)
    data = Data(x = features.float(), edge_index = edge_index, edge_type = edge_type, y = label)

    print("loading index...")
    data.train_idx = torch.load(path+'train_idx.pt') + review0
    data.n_id = torch.arange(data.num_nodes)
    data.valid_idx = torch.load(path+'val_idx.pt') + review0
    data.test_idx = torch.load(path+'test_idx.pt') + review0

    return data

data = load_data() # load data

train_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=data.train_idx, batch_size=args.BATCH_SIZE, shuffle=True)
test_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=data.test_idx, batch_size=args.BATCH_SIZE)
val_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=data.valid_idx, batch_size=args.BATCH_SIZE)

# model = MVSD(in_dim=args.IN_DIM, hidden_dim=args.HIDDEN_DIM, meta_hidden=args.META_HIDDEN, out_dim=args.OUT_DIM, dropout=args.DROPOUT)
model = MVSD(in_dim=args.IN_DIM, hidden_dim=args.HIDDEN_DIM, dropout=args.DROPOUT, sd_layer=args.N_SD, n_layer=args.N_LAYER, c_c=args.c_c, int_type=args.int_type, conv_type=args.conv)
#print(model)
model.to(device)
model.apply(init_weights)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([{'params' : model.parameters()},
                               {'params': year1},
                               {'params': year2},
                               {'params': rating1},
                               {'params': rating2},
                               {'params': rating_meta},
                               {'params': year_meta}
                               ], lr = args.LEARNING_RATE, weight_decay = args.WEIGHT_DECAY, amsgrad=True)

scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.7,patience=args.patience,threshold=5e-5)

def train():
    model.train()
    total_examples = total_loss = 0
    W1 = W2 = W3 = W4 = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size      
        out = model(batch)[:batch_size]
        y = batch.y[batch.n_id[:batch_size] - review0]
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
    W1 = W1 * args.BATCH_SIZE / total_examples 
    W2 = W2 * args.BATCH_SIZE / total_examples
    W3 = W3 * args.BATCH_SIZE / total_examples
    W4 = W4 * args.BATCH_SIZE / total_examples 
    return total_loss / total_examples, W1, W2, W3, W4

@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_correct = total_loss = 0
    W1 = W2 = W3 = W4 = 0
    for batch in tqdm(loader):      
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch)[:batch_size]
        y = batch.y[batch.n_id[:batch_size] - review0]
        review_id.append(batch.n_id[:batch_size].cpu() - review0)
        loss = loss_fn(out, y)
        label.append(y.cpu())
        pred = out.argmax(dim=-1)
        y_predict.append(pred.cpu())
        total_examples += batch_size
        total_correct += int((pred == y).sum())
        total_loss += float(loss) * batch_size
    W1 = W1 * args.BATCH_SIZE / total_examples 
    W2 = W2 * args.BATCH_SIZE / total_examples
    W3 = W3 * args.BATCH_SIZE / total_examples
    W4 = W4 * args.BATCH_SIZE / total_examples
    return total_correct / total_examples, total_loss / total_examples, W1, W2, W3, W4

print("begin to train...")

# set time
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
log_dir="./log/fit/" + bj_dt.now().strftime("%Y%m%d-%H%M%S")
dt = bj_dt.now().strftime("%Y%m%d-%H%M%S")

max_acc = 0

for epoch in range(args.EPOCH):
    time = bj_dt.now().strftime("%Y%m%d-%H:%M:%S")
    print(f"Epoch {epoch}\n---------{time}---------------")
    loss, W1, W2, W3, W4 = train()
    y_predict, label, review_id = [], [], []
    #print(f'Train Weight: {float(W1):.5f} {float(W2):.5f} {float(W3):.5f} {float(W4):.5f}')
    val_acc, val_loss, W1, W2, W3, W4 = test(val_loader)
    #scheduler.step(val_loss)
    print(f'Loss: {loss:.5f}, Val_loss: {val_loss:.5f}, Val_acc: {val_acc:.5f}')
    if val_acc > max_acc:
        max_acc = val_acc
        y_predict, label, review_id = [], [], []
        test_acc, test_loss, W1, W2, W3, W4 = test(test_loader)
        y_predict = np.concatenate(y_predict)
        label = np.concatenate(label)
        review_id = np.concatenate(review_id)
        np.save("./res/y_predict.npy",y_predict)
        np.save("./res/label.npy",label)
        np.save("./res/review_id.npy",review_id)
        f1 = f1_score(label, y_predict)
        AUC = auc(label, y_predict)
        # to attain a test metric just in case stopping halfway, as it might take long to train
        print(f"max_test_acc: {test_acc:.5f}  F1: {f1:.5f}  AUC: {AUC:.5f}")
        torch.save(model.state_dict(),'./models/'+dt+'.pt')

model.load_state_dict(torch.load('./models/'+dt+'.pt'))
test_acc, _, _, _, _, _ = test(test_loader)
print("test:")
print(f"test_acc: {test_acc:.5f}  F1: {f1:.5f}  AUC: {AUC:.5f}")


