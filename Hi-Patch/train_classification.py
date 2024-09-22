# -*- coding:utf-8 -*-
import os
import argparse
import warnings
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='physionet', choices=['P12', 'P19', 'physionet', 'mimic3'])
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--epochs', type=int, default=10)  #
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--history', type=int, default=48, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('-ps', '--patch_size', type=float, default=6, help="window size for a patch")
parser.add_argument('--stride', type=float, default=6, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('--alpha', type=float, default=1, help="Uncertainty base number")
parser.add_argument('--res', type=float, default=1, help="Res")


args, unknown = parser.parse_known_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from model import *
from model.hipatch import *
from lib.utils import *

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Create model save path
model_path = './models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Load command line hyperparameters
dataset = args.dataset
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs


def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)

print('Dataset used: ', dataset)

# Set dataset parameters
if dataset == 'P12':
    base_path = '../data/P12'
    start = 0
    variables_num = 36
    d_static = 9
    args.d_static = 9
    timestamp_num = 215
    n_class = 2
    args.n_class = 2
    split_idx = 1
    args.history = 48
elif dataset == 'physionet':
    base_path = '../data/physionet'
    start = 4
    variables_num = 36
    d_static = 9
    args.d_static = 9
    timestamp_num = 215
    n_class = 2
    args.n_class = 2
    split_idx = 5
    args.history = 48
elif dataset == 'P19':
    base_path = '../data/P19'
    d_static = 6
    args.d_static = 6
    variables_num = 34
    timestamp_num = 60
    n_class = 2
    args.n_class = 2
    split_idx = 1
    args.history = 60
elif dataset == 'mimic3':
    base_path = '../data/mimic3'
    start = 0
    d_static = 0
    args.d_static = 0
    variables_num = 16
    timestamp_num = 292
    n_class = 2
    args.n_class = 2
    split_idx = 0
    args.history = 48

# Evaluation metrics
acc_arr = []
auprc_arr = []
auroc_arr = []
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Run five experiments
for k in range(5):
    # Set different random seed
    torch.manual_seed(k)
    torch.cuda.manual_seed(k)
    np.random.seed(k)

    # Load semantic representations of variables obtained through PLM
    if dataset == 'P12':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'physionet':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
    elif dataset == 'mimic3':
        split_path = ''

    # Prepare data and split the dataset
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    args.ndim = variables_num
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / args.history
    args.task = 'classification'

    # Normalize data and extract required model inputs
    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_extract_feature_patch(Ptrain, ytrain, mf, stdf, ms, ss, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_extract_feature_patch(Pval, yval, mf, stdf, ms, ss, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_extract_feature_patch(Ptest, ytest, mf, stdf, ms, ss, args)

    elif dataset == 'mimic3':
        T, F = timestamp_num, variables_num
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i][:Ptrain[i][4]] = Ptrain[i][2]

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Ptrain, ytrain, mf, stdf, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Pval, yval, mf, stdf, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Ptest, ytest, mf, stdf, args)

    # Load the model
    model = Hi_Patch(args).to(args.device)

    params = (list(model.parameters()))
    print('model', model)
    print('parameters:', count_parameters(model))

    # Cross-entropy loss, Adam optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Upsample minority class
    idx_0 = np.where(ytrain == 0)[0]
    idx_1 = np.where(ytrain == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)
    K0 = n0 // int(batch_size / 2)
    K1 = expanded_n1 // int(batch_size / 2)
    n_batches = np.min([K0, K1])

    best_val_epoch = 0
    best_aupr_val = best_auc_val = 0.0
    best_loss_val = 100.0

    print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (
        num_epochs, n_batches, num_epochs * n_batches))

    start = time.time()

    for epoch in range(num_epochs):
        if epoch - best_val_epoch > 5:
            break
        """Training"""
        model.train()

        # Shuffle data
        np.random.shuffle(expanded_idx_1)
        I1 = expanded_idx_1
        np.random.shuffle(idx_0)
        I0 = idx_0
        for n in range(n_batches):
            # Get current batch data
            idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
            P, P_mask, P_static, P_time, y = \
                Ptrain_tensor[idx].cuda(), Ptrain_mask_tensor[idx].cuda(), Ptrain_static_tensor[idx].cuda() if d_static != 0 else None, \
                    Ptrain_time_tensor[idx].cuda(), ytrain_tensor[idx].cuda()

            # Backward pass
            outputs = model.classification(P, P_time, P_mask, P_static)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Calculate training set evaluation metrics
        train_probs = torch.squeeze(torch.sigmoid(outputs))
        train_probs = train_probs.cpu().detach().numpy()
        train_y = y.cpu().detach().numpy()
        train_auroc = roc_auc_score(train_y, train_probs[:, 1])
        train_auprc = average_precision_score(train_y, train_probs[:, 1])

        """Validation"""
        model.eval()
        with torch.no_grad():
            out_val = evaluate_model_patch(model, Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor,
                                        n_classes=n_class, batch_size=batch_size)
            out_val = torch.squeeze(torch.sigmoid(out_val))
            out_val = out_val.detach().cpu().numpy()
            y_val_pred = np.argmax(out_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]
            val_loss = torch.nn.CrossEntropyLoss().cuda()(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())
            auc_val = roc_auc_score(yval, out_val[:, 1])
            aupr_val = average_precision_score(yval, out_val[:, 1])
            print(
                "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                 val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))

            # Save the model weights with the best AUPRC on the validation set
            if aupr_val > best_aupr_val:
                best_auc_val = auc_val
                best_aupr_val = aupr_val
                best_val_epoch = epoch
                save_time = str(int(time.time()))
                torch.save(model.state_dict(),
                           model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt')


            out_test = evaluate_model_patch(model, Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor,
                                            Ptest_time_tensor,
                                            n_classes=n_class, batch_size=batch_size).numpy()
            denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
            y_test = ytest.copy()
            probs = np.exp(out_test.astype(np.float64)) / denoms
            ypred = np.argmax(out_test, axis=1)
            acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]
            auc = roc_auc_score(y_test, probs[:, 1])
            aupr = average_precision_score(y_test, probs[:, 1])

            print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))




    end = time.time()
    time_elapsed = end - start
    print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

    """testing"""
    model.eval()
    model.load_state_dict(
        torch.load(model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'))
    with torch.no_grad():
        out_test = evaluate_model_patch(model, Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor,
                                        n_classes=n_class, batch_size=batch_size).numpy()
        denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
        y_test = ytest.copy()
        probs = np.exp(out_test.astype(np.float64)) / denoms
        ypred = np.argmax(out_test, axis=1)
        acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]
        auc = roc_auc_score(y_test, probs[:, 1])
        aupr = average_precision_score(y_test, probs[:, 1])

        print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
        print('classification report', classification_report(y_test, ypred))
        print(confusion_matrix(y_test, ypred, labels=list(range(n_class))))

    acc_arr.append(acc * 100)
    auprc_arr.append(aupr * 100)
    auroc_arr.append(auc * 100)

print('args.dataset', args.dataset)
# Display the mean and standard deviation of five runs
mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
print('------------------------------------------')
print('Accuracy = %.1f±%.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f±%.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f±%.1f' % (mean_auroc, std_auroc))