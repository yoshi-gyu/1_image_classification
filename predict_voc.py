# パッケージのimport
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

# from torchvision import models
from model import Model

from tqdm import tqdm

from utils.dataloader_voc import make_datapath_list, DataTransform, Anno_xml2list, VOCDataset

# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, alpha=0.1):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0.0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print('inputs:', inputs.size())
                # print('labels:', labels.size())

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    # print('outputs:', outputs.size())

                    # print('outputs:', outputs)
                    # print('labels:', labels)

                    loss = criterion(outputs, labels)   # 損失を計算
                    # print('loss:', loss.data)
                    # loss_reg = (torch.tensor(0.0)).to(device)
                    # for output in outputs:
                    #     loss_reg += torch.norm(output)
                    # print('loss:', loss.data, 'loss_reg:', loss_reg.data)
                    # loss += alpha * loss_reg


                    # _, preds = torch.max(outputs, 1)  # ラベルを予測
                    preds = torch.sigmoid(outputs)

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    # epoch_corrects += torch.sum(preds == labels.data)
                    for pred, label in zip(preds, labels):
                        epoch_corrects += torch.dot(pred, label.data.float())
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


if __name__ == '__main__':
    # 乱数のシードを設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # ファイルパスのリストを作成
    rootpath = "/home/yoshi/Project/pytorch_advanced/data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath)

    # # 動作確認
    # print(train_img_list[0])

    # VOCクラス名　
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']
    num_classes = len(voc_classes)

    # VOCDataset動作確認
    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    # input_size = 300  # 画像のinputサイズを300×300にする
    voc_size = 224

    train_dataset = VOCDataset(num_classes, train_img_list, train_anno_list, phase="train", transform=DataTransform(
        voc_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    val_dataset = VOCDataset(num_classes, val_img_list, val_anno_list, phase="val", transform=DataTransform(
        voc_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

    # データローダーの作成
    batch_size = 8 #4
    # train_dataloader = data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataloader = data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # # 学習済みのVGG-19モデルをロード

    # # VGG-19モデルのインスタンスを生成
    # use_pretrained = True  # 学習済みのパラメータを使用
    # net = models.vgg19(pretrained=use_pretrained)

    # # VGG19の最後の出力層の出力ユニットをVOCの20クラスに付け替える
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    net = Model(num_classes)

    # 訓練モードに設定
    net.train()

    print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

    # 損失関数の設定
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # ファインチューニングで学習させるパラメータを、変数params_to_updateの1～3に格納する
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    # 学習させる層のパラメータ名を指定
    update_param_names_1 = ["features"]
    update_param_names_2 = ["vgg.classifier.0.weight",
                            "vgg.classifier.0.bias", "vgg.classifier.3.weight", "vgg.classifier.3.bias"]
    update_param_names_3 = ["vgg.classifier.6.weight", "vgg.classifier.6.bias"]

    # パラメータごとに各リストに格納する
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            print("params_to_update_1に格納：", name)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            print("params_to_update_2に格納：", name)

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            print("params_to_update_3に格納：", name)

        else:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)

    # 最適化手法の設定
    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
    ], momentum=0.9)

    # 学習・検証を実行する
    num_epochs=20
    alpha = 1.0
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, alpha=alpha)

    # PyTorchのネットワークパラメータの保存
    save_path = './weights_fine_tuning_alpha{}_ep{}.pth'.format(alpha, num_epochs)
    torch.save(net.state_dict(), save_path)