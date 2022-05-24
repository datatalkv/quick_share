import numpy as np
import pandas as pd
import torch

from stml_mft_china_eq.pyutils import misc_utils
from stml_mft_china_eq.pyutils.tqdmX import TqdmWrapper, format_str
from stml_mft_china_eq.statarb.gym.predictors import layer_nn
from stml_mft_china_eq.statarb.gym.predictors import loss_nn
from stml_mft_china_eq.statarb.gym.predictors import utils_nn
from stml_mft_china_eq.statarb.gym.predictors.predictor_nn import MLP


def test_MLP():
    SNR, target_std = 0.1, 0.1
    n_sample, ratio_train = 10000, 0.5
    n_in, n_out = 20, 1
    # n_hiddens = [100, 50, 30]
    n_hiddens = [50, 30, 5]
    x = torch.randn(n_sample, n_in)
    m_gen = MLP(n_in=n_in, n_out=n_out, n_hiddens=n_hiddens, bias=True, activation="mish", dropout=0, use_bucket_emb=False)
    # m_gen = MLP(n_in=n_in, n_out=n_out, n_hiddens=n_hiddens, bias=True, activation="mish", dropout=0, use_bucket_emb=True)
    m_gen.eval()
    with torch.no_grad():
        y_true = m_gen(x)
        y = (y_true - y_true.mean()) / y_true.std()
        y = y * SNR + torch.randn(n_sample, n_out) + 0.1
    n_sample_train = int(n_sample * ratio_train)
    n_sample_test = n_sample - n_sample_train
    x_train, y_train = x[:n_sample_train], y[:n_sample_train]
    x_test,  y_test  = x[n_sample_train:], y[n_sample_train:]

    batch_size = 2000
    num_workers = 1
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_train_eval = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(misc_utils.get_gpu_device()))
        _just_take_position = torch.from_numpy(np.random.randn(100)).float().to(device)
    else:
        device = torch.device("cpu")

    list_stats_y_train, list_stats_yhat_train, list_regeval_train, list_regeval_pos_train, list_regeval_neg_train = {}, {}, {}, {}, {}
    list_stats_y_test, list_stats_yhat_test, list_regeval_test, list_regeval_pos_test, list_regeval_neg_test = {}, {}, {}, {}, {}
    for use_bucket_emb in [False, True]:
        for use_was in [False]:
            for bias in [True, False]:
                for activation in ["mish", "tanh"]:
                    # for dropout in [None, 0.5]:
                    for dropout in [0.5]:
                        for bool_flip_aug in [False]:
                            for loss_type in ["l1", "l2", "huber"]:
                                for is_focal in [True, False]:
                                    for is_poly1 in [True, False]:
                                        for reduction in ["mean", "robust"]:
                                            # loss_function = loss_nn.WeightedLoss(loss_type=loss_type, is_focal=is_focal, is_poly1=is_poly1, reduction=reduction)
                                            loss_function = loss_nn.CorrRWas(was_target_mean=0.0, was_target_std=target_std)
                                            loss_was_function = loss_nn.WassersteinNormalUpper(target_mean=0.0, target_std=target_std)
                                            m = MLP(n_in=n_in, n_out=n_out, n_hiddens=n_hiddens,
                                                    bias=bias, activation=activation, dropout=dropout, use_bucket_emb=use_bucket_emb).to(device)
                                            current_model_key = f"{m.str_name}_was{use_was}_flip{bool_flip_aug}_loss{loss_function.str_name}"
                                            # warmup prior
                                            optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)
                                            for i_epoch in range(10):
                                                with TqdmWrapper(dataloader_train) as tw:
                                                    tw.set_description(f"WarmupEp {i_epoch} {current_model_key}")
                                                    for i_batch, (batch_x_train, _) in enumerate(TqdmWrapper(dataloader_train)):
                                                        batch_x_train = batch_x_train.to(device)
                                                        optimizer.zero_grad()
                                                        batch_yhat_train = m(batch_x_train)
                                                        loss_em_upper = loss_was_function(batch_yhat_train)
                                                        loss_em_upper.backward()
                                                        optimizer.step()
                                                    stats_y_train, stats_yhat_train, regeval_train, regeval_pos_train, regeval_neg_train = utils_nn.evaluate_net(m, dataloader_train_eval)
                                                    stats_y_test, stats_yhat_test, regeval_test, regeval_pos_test, regeval_neg_test =  utils_nn.evaluate_net(m, dataloader_test)

                                                    tw.add({
                                                        "warmup_stats_train": "\n{}".format(pd.concat([stats_y_train, stats_yhat_train], axis=0)),
                                                        "warmup_eval_train": "\n{}".format(pd.concat([regeval_train, regeval_pos_train, regeval_neg_train], axis=0))
                                                    }, kv_format=format_str(['bold', 'blue'], '{key} : ') + '{value}')
                                                    tw.add({
                                                        "warmup_stats_test": "\n{}".format(pd.concat([stats_y_test, stats_yhat_test], axis=0)),
                                                        "warmup_eval_test": "\n{}".format(pd.concat([regeval_test, regeval_pos_test, regeval_neg_test], axis=0))
                                                    }, kv_format=format_str(['bold', 'red'], '{key} : ') + '{value}')
                                                    tw.update()

                                            # real training
                                            optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)
                                            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
                                            for i_epoch in range(10, 40):
                                                stats_y_train_this_epoch, stats_yhat_train_this_epoch = [], []
                                                regeval_train_this_epoch, regeval_train_pos_this_epoch, regeval_train_neg_this_epoch = [], [], []
                                                with TqdmWrapper(dataloader_train) as tw:
                                                    tw.set_description(f"Ep {i_epoch} lr={lr_scheduler.get_last_lr()[0]:.5f} {current_model_key}")
                                                    for i_batch, (batch_x_train, batch_y_train) in enumerate(TqdmWrapper(dataloader_train)):
                                                        if bool_flip_aug:
                                                            batch_x_train, batch_y_train = torch.cat([batch_x_train, -batch_x_train], 0).to(device), torch.cat([batch_y_train, -batch_y_train], 0).to(device)
                                                        else:
                                                            batch_x_train, batch_y_train = batch_x_train.to(device), batch_y_train.to(device)

                                                        # training data
                                                        optimizer.zero_grad()
                                                        batch_yhat_train = m(batch_x_train)
                                                        mse_loss = loss_function(
                                                            batch_yhat_train.view([-1]),
                                                            batch_y_train.view([-1]),
                                                            weights=torch.ones_like(batch_yhat_train.view([-1])).to(device)
                                                        )
                                                        if use_was:
                                                            mse_loss += loss_was_function(batch_yhat_train)
                                                        if use_bucket_emb:
                                                            mse_loss += m.l_aux
                                                        mse_loss.backward()
                                                        optimizer.step()

                                                    stats_y_train, stats_yhat_train, regeval_train, regeval_pos_train, regeval_neg_train = utils_nn.evaluate_net(m, dataloader_train_eval)
                                                    stats_y_test, stats_yhat_test, regeval_test, regeval_pos_test, regeval_neg_test = utils_nn.evaluate_net(m, dataloader_test)

                                                    tw.add({
                                                        "stats_train": "\n{}".format(pd.concat([stats_y_train, stats_yhat_train], axis=0)),
                                                        "eval_train": "\n{}".format(pd.concat([regeval_train, regeval_pos_train, regeval_neg_train], axis=0))
                                                    }, kv_format=format_str(['bold', 'blue'], '{key} : ') + '{value}')
                                                    tw.add({
                                                        "stats_test": "\n{}".format(pd.concat([stats_y_test, stats_yhat_test], axis=0)),
                                                        "eval_test": "\n{}".format(pd.concat([regeval_test, regeval_pos_test, regeval_neg_test], axis=0))
                                                    }, kv_format=format_str(['bold', 'red'], '{key} : ') + '{value}')
                                                    tw.update()
                                                lr_scheduler.step()


                                            stats_y_train, stats_yhat_train, regeval_train, regeval_pos_train, regeval_neg_train = utils_nn.evaluate_net(m, dataloader_train_eval)
                                            stats_y_test, stats_yhat_test, regeval_test, regeval_pos_test, regeval_neg_test = utils_nn.evaluate_net(m, dataloader_test)
                                            list_stats_y_train[current_model_key] = stats_y_train
                                            list_stats_yhat_train[current_model_key] = stats_yhat_train
                                            list_regeval_train[current_model_key] = regeval_train
                                            list_regeval_pos_train[current_model_key] = regeval_pos_train
                                            list_regeval_neg_train[current_model_key] = regeval_neg_train
                                            list_stats_y_test[current_model_key] = stats_y_test
                                            list_stats_yhat_test[current_model_key] = stats_yhat_test
                                            list_regeval_test[current_model_key] = regeval_test
                                            list_regeval_pos_test[current_model_key] = regeval_pos_test
                                            list_regeval_neg_test[current_model_key] = regeval_neg_test

    arr = []
    for k in list_regeval_test.keys():
        v1 = list_regeval_test[k]
        v1.index = [k]
        v2 = list_stats_yhat_test[k]
        v2.index = [k]
        arr.append(pd.concat([v1, v2], axis=1))
    df = pd.concat(arr, axis=0)
    import sys,IPython; IPython.embed(); sys.exit(0)


if __name__ == "__main__":
    test_MLP()
