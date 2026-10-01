import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from layers import MultiLossLayer
from experiment_tracking import ExperimentTracker
from loss import ial_loss, icl_loss
from memory_utils import (
    backward_through_gradient_bridge,
    checkpoint_loss,
    gradient_proxy,
)
from model import MIEstimator, MultiModalEncoder, list_rebul_sort
from utils import (
    csls_sim,
    get_adjr,
    get_ids,
    load_att_text,
    load_attr,
    load_img,
    load_rel_text,
    load_relation,
    pairwise_distances,
    read_raw_data,
)

best_result = None
best_mrr = 0.0
top_k = [1, 5, 10]


def load_img_features(ent_num, file_dir):
    # load images features
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = (
            "data/mmkb-datasets/"
            + filename
            + "/"
            + filename
            + "_id_img_feature_dict.pkl"
        )
    else:
        img_vec_path = None

    img_features = load_img(ent_num, img_vec_path)
    return img_features


def load_att_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        att_text_vec_path = (
            "data/mmkb-datasets/" + filename + "/" + "attribute_feature_dict.pkl"
        )  # roberta-base
        att_txt_features = load_att_text(ent_num, att_text_vec_path)
    else:
        att_txt_features = None
    return att_txt_features


def load_rel_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        rel_text_vec_path = (
            "data/mmkb-datasets/" + filename + "/" + "triples_feature_dict.pkl"
        )
    else:
        pass

    rel_txt_features = load_rel_text(ent_num, rel_text_vec_path)
    return rel_txt_features


class HUMEA:
    def __init__(self, args=None, tracker=None):
        self.parser = argparse.ArgumentParser()
        self.args = args if args is not None else self.parse_options(self.parser)
        self.tracker = tracker
        self.set_seed(self.args.seed, True)
        self.device = torch.device(self.args.device)
        self.init_data()
        self.init_model()

    @staticmethod
    def parse_options(parser):
        parser.add_argument(
            "--file_dir",
            type=str,
            default="data/DBP15K/zh_en",
            required=False,
        )
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument(
            "--epochs", type=int, default=1000, help="number of epochs to train"
        )
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument(
            "--hidden_units",
            type=str,
            default="300,300,300",
        )
        parser.add_argument(
            "--heads",
            type=str,
            default="2,2",
        )
        parser.add_argument(
            "--instance_normalization",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--lr", type=float, default=0.005, help="initial learning rate"
        )

        parser.add_argument(
            "--dropout", type=float, default=0.0, help="dropout rate for layers"
        )
        parser.add_argument(
            "--attn_dropout",
            type=float,
            default=0.0,
            help="dropout rate for gat layers",
        )
        parser.add_argument(
            "--gph_dim",
            type=int,
            default=300,
        )
        parser.add_argument(
            "--n_exp",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--topk",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--without",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--csls", action="store_true", default=False, help="use CSLS for inference"
        )
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument(
            "--il_start", type=int, default=500, help="If Il, when to start?"
        )
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument(
            "--tau_cl",
            type=float,
            default=0.1,
            help="the temperature factor of contrastive loss",
        )
        parser.add_argument(
            "--tau_al",
            type=float,
            default=1,
            help="the temperature factor of alignment loss",
        )
        parser.add_argument(
            "--feat_dim", type=int, default=300, help="the hidden size of img feature"
        )
        parser.add_argument(
            "--use_project_head",
            action="store_true",
            default=False,
            help="use projection head",
        )

        parser.add_argument(
            "--al_loss", type=float, default=0.1, help="narrow the range of losses"
        )
        parser.add_argument(
            "--cl_loss", type=float, default=1, help="narrow the range of losses"
        )
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--train_ill_path", type=str, default="", help="")
        parser.add_argument(
            "--fusion_weight_dim",
            type=int,
            default=0,
            help="fusion_weight_dim",
        )
        parser.add_argument(
            "--mi_loss", type=float, default=0.0001, help="the weight of NTXent Loss"
        )
        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        filename = os.path.split(file_dir)[-1].upper()
        self.train_name = f"{filename}_{self.args.rate}"
        device = self.device
        self.ent2id_dict, self.ills, self.triples = read_raw_data(file_dir, lang_list)
        left_ents = get_ids(os.path.join(file_dir, "ent_ids_1"))
        right_ents = get_ids(os.path.join(file_dir, "ent_ids_2"))
        self.ENT_NUM = len(self.ent2id_dict)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{self.train_name}_{timestamp}.log"
        self.training_log_path = Path("log") / log_file
        logger.add("log/" + log_file)
        np.random.shuffle(self.ills)
        self.img_features = F.normalize(
            torch.Tensor(load_img_features(self.ENT_NUM, file_dir)).to(device)
        )
        self.att_txt_features = F.normalize(
            torch.Tensor(load_att_txt_features(self.ENT_NUM, file_dir)).to(device)
        )
        self.rel_txt_features = F.normalize(
            torch.Tensor(load_rel_txt_features(self.ENT_NUM, file_dir)).to(device)
        )
        self.adj = get_adjr(self.ENT_NUM, self.triples, norm=True).to(self.device)
        self.rel_features = torch.Tensor(
            load_relation(self.ENT_NUM, self.triples, 1000)
        ).to(device)

        self.att_features = torch.Tensor(
            load_attr(
                [
                    os.path.join(file_dir, "training_attrs_1"),
                    os.path.join(file_dir, "training_attrs_2"),
                ],
                self.ENT_NUM,
                self.ent2id_dict,
                1000,
            )
        ).to(device)
        # train/val/test split
        if self.args.train_ill_path:
            logger.info(f"use {self.args.train_ill_path}")
            self.train_ill = np.load(self.args.train_ill_path)
        else:
            self.train_ill = np.array(
                self.ills[: int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32
            )

        self.test_ill = np.array(
            self.ills[int(len(self.ills) // 1 * self.args.rate) :], dtype=np.int32
        )

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)
        self.left_non_train = list(set(left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(
            set(right_ents) - set(self.train_ill[:, 1].tolist())
        )

    def init_model(self):
        self.mi_estimator = MIEstimator(self.args).to(self.device)
        self.multimodal_encoder = MultiModalEncoder(
            args=self.args,
            ent_num=self.ENT_NUM,
        ).to(self.device)
        self.multi_loss_layer = MultiLossLayer(loss_num=6).to(self.device)
        self.align_multi_loss_layer = MultiLossLayer(loss_num=6).to(self.device)
        self.params = [
            {
                "params": list(self.multimodal_encoder.parameters())
                + list(self.multi_loss_layer.parameters())
                + list(self.align_multi_loss_layer.parameters())
            }
        ]

        self.optimizer = optim.AdamW(self.params, lr=self.args.lr)
        self.mi_optimizer = optim.AdamW(
            [{"params": list(self.mi_estimator.parameters())}], lr=self.args.lr
        )
        self.criterion_cl = icl_loss(
            device=self.device,
            tau=self.args.tau_cl,
            ab_weight=0.5,
            n_view=2,
        )
        self.criterion_align = ial_loss(
            device=self.device,
            tau=self.args.tau_al,
            ab_weight=0.5,
            zoom=self.args.al_loss,
            reduction=self.args.reduction,
        )

    def semi_supervised_learning(self):
        with torch.no_grad():
            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                att_text_features=self.att_txt_features,
                rel_text_features=self.rel_txt_features,
            )

            final_emb = F.normalize(joint_emb)

        distance_list = []
        for i in np.arange(0, len(self.left_non_train), 1000):
            d = pairwise_distances(
                final_emb[self.left_non_train[i : i + 1000]],
                final_emb[self.right_non_train],
            )
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        return preds_l, preds_r

    def inner_view_loss(
        self, gph_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, img_emb, train_ill
    ):
        zoom = self.args.cl_loss
        loss_GCN = checkpoint_loss(self.criterion_cl, gph_emb, train_ill)
        loss_rel = checkpoint_loss(self.criterion_cl, rel_emb, train_ill)
        loss_att = checkpoint_loss(self.criterion_cl, att_emb, train_ill)
        loss_img = checkpoint_loss(self.criterion_cl, img_emb, train_ill)
        loss_att_text = checkpoint_loss(self.criterion_cl, att_text_emb, train_ill)
        loss_rel_text = checkpoint_loss(self.criterion_cl, rel_text_emb, train_ill)

        total_loss = (
            self.multi_loss_layer(
                [loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img]
            )
            * zoom
        )
        return total_loss

    def kl_alignment_loss(
        self,
        joint_emb,
        gph_emb,
        rel_emb,
        att_emb,
        att_text_emb,
        rel_text_emb,
        img_emb,
        train_ill,
    ):
        zoom = self.args.al_loss
        loss_GCN = checkpoint_loss(
            self.criterion_align, gph_emb, joint_emb, train_ill
        )
        loss_rel = checkpoint_loss(
            self.criterion_align, rel_emb, joint_emb, train_ill
        )
        loss_att = checkpoint_loss(
            self.criterion_align, att_emb, joint_emb, train_ill
        )
        loss_img = checkpoint_loss(
            self.criterion_align, img_emb, joint_emb, train_ill
        )
        loss_att_text = checkpoint_loss(
            self.criterion_align, att_text_emb, joint_emb, train_ill
        )
        loss_rel_text = checkpoint_loss(
            self.criterion_align, rel_text_emb, joint_emb, train_ill
        )

        total_loss = (
            self.align_multi_loss_layer(
                [loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img]
            )
            * zoom
        )
        return total_loss

    def train(self):
        logger.info(f"{self.args}")
        t_total = time.time()
        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)
        for epoch in range(0, self.args.epochs):
            epoch_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            self.multimodal_encoder.train()
            self.multi_loss_layer.train()
            self.align_multi_loss_layer.train()
            self.mi_estimator.eval()
            self.optimizer.zero_grad()

            (
                [
                    gph_emb,
                    img_emb,
                    rel_emb,
                    att_emb,
                    att_text_emb,
                    rel_text_emb,
                    joint_emb,
                ],
                embeddings,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                self.att_txt_features,
                self.rel_txt_features,
                exp_outputs=True,
            )
            encoder_outputs = [
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ]
            encoder_proxies = [gradient_proxy(output) for output in encoder_outputs]
            (
                gph_loss_emb,
                img_loss_emb,
                rel_loss_emb,
                att_loss_emb,
                att_text_loss_emb,
                rel_text_loss_emb,
                joint_loss_emb,
            ) = encoder_proxies
            expert_outputs = [
                output for outputs in embeddings.values() for output in outputs
            ]
            proxy_embeddings = {
                key: [gradient_proxy(output) for output in outputs]
                for key, outputs in embeddings.items()
            }
            expert_proxies = [
                output for outputs in proxy_embeddings.values() for output in outputs
            ]
            loss_all = [self.args.mi_loss * self.mi_estimator(proxy_embeddings)]
            epoch_losses = {"mi": float(loss_all[0].detach()), "inner": 0.0,
                            "alignment": 0.0, "joint": 0.0}
            np.random.shuffle(self.train_ill)
            if epoch <= self.args.il_start:
                if epoch % 50 == 0:
                    k_o = 5
                    k_v = k_o - epoch // 50
                    if k_v < 1:
                        self.train_list = self.train_ill
                    else:
                        self.train_list = list_rebul_sort(
                            self.train_ill, joint_emb, k=k_v
                        )
                        print("K value is :" + str(k_v))
            else:
                self.train_list = self.train_ill

            for si in np.arange(0, self.train_list.shape[0], bsize):
                if self.args.without != 7:
                    # single model align
                    in_loss = self.inner_view_loss(
                        gph_loss_emb,
                        rel_loss_emb,
                        att_loss_emb,
                        att_text_loss_emb,
                        rel_text_loss_emb,
                        img_loss_emb,
                        self.train_list[si : si + bsize],
                    )
                    loss_all.append(in_loss)
                    epoch_losses["inner"] += float(in_loss.detach())

                if self.args.without != 8:
                    # joint align to single
                    in_loss = self.kl_alignment_loss(
                        joint_loss_emb,
                        gph_loss_emb,
                        rel_loss_emb,
                        att_loss_emb,
                        att_text_loss_emb,
                        rel_text_loss_emb,
                        img_loss_emb,
                        self.train_list[si : si + bsize],
                    )
                    loss_all.append(in_loss)
                    epoch_losses["alignment"] += float(in_loss.detach())

            torch.cuda.empty_cache()
            in_loss = loss_joi = None
            for si in np.arange(0, self.train_list.shape[0], bsize):
                loss_joi = checkpoint_loss(
                    self.criterion_cl,
                    joint_loss_emb,
                    self.train_list[si : si + bsize],
                )
                loss_all.append(loss_joi)
                epoch_losses["joint"] += float(loss_joi.detach())

            epoch_losses["total"] = float(sum(loss_all).detach())
            torch.cuda.empty_cache()
            sum(loss_all).backward()
            bridge_outputs = encoder_outputs + expert_outputs
            bridge_proxies = encoder_proxies + expert_proxies
            del loss_all, proxy_embeddings, in_loss, loss_joi
            torch.cuda.empty_cache()
            backward_through_gradient_bridge(bridge_outputs, bridge_proxies)
            self.optimizer.step()

            del (
                bridge_outputs,
                bridge_proxies,
                encoder_outputs,
                encoder_proxies,
                expert_outputs,
                expert_proxies,
                embeddings,
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
                gph_loss_emb,
                img_loss_emb,
                rel_loss_emb,
                att_loss_emb,
                att_text_loss_emb,
                rel_text_loss_emb,
                joint_loss_emb,
            )
            torch.cuda.empty_cache()

            # train estimator
            self.mi_estimator.train()
            self.mi_optimizer.zero_grad()
            with torch.no_grad():
                _, embeddings = self.multimodal_encoder(
                    self.device,
                    self.input_idx,
                    self.adj,
                    self.img_features,
                    self.rel_features,
                    self.att_features,
                    self.att_txt_features,
                    self.rel_txt_features,
                    exp_outputs=True,
                )
            estimator_loss = self.mi_estimator.train_estimator(embeddings)
            estimator_loss.backward()
            self.mi_optimizer.step()
            epoch_losses["estimator"] = float(estimator_loss.detach())
            epoch_losses["seconds"] = time.monotonic() - epoch_started
            if device.type == "cuda":
                epoch_losses["gpu_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
            if self.tracker is not None:
                self.tracker.log_losses(epoch, epoch_losses)

            self.optimizer.zero_grad(set_to_none=True)
            self.mi_optimizer.zero_grad(set_to_none=True)
            del embeddings, estimator_loss, _
            torch.cuda.empty_cache()

            if epoch != 0 and epoch % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)

        print("[optimization finished!]")
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

    def test(self, epoch):
        global best_mrr, best_result
        with torch.no_grad():
            self.multimodal_encoder.eval()
            self.multi_loss_layer.eval()
            self.align_multi_loss_layer.eval()

            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                att_text_features=self.att_txt_features,
                rel_text_features=self.rel_txt_features,
            )
            logger.info(f"{self.multimodal_encoder.joint.latest_weights}")
            results = {
                # "gph_emb": gph_emb,
                # "img_emb": img_emb,
                # "att_emb": att_emb,
                "joint_emb": joint_emb,
                # "rel_emb": rel_emb,
                # "att_text_emb": att_text_emb,
                # "rel_text_emb": rel_text_emb,
            }
            for name, emb in results.items():
                acc, mr, mrr = self.evaluate_embedding(emb, f"epoch {epoch} - {name}")
                if self.tracker is not None:
                    self.tracker.log_evaluation(epoch, {
                        "hits1": float(acc[0]), "hits5": float(acc[1]),
                        "hits10": float(acc[2]), "mr": float(mr), "mrr": float(mrr),
                    })
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_result = (epoch, name, acc, mr, mrr)

            del (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            )

    def evaluate_embedding(self, emb, name):
        emb = F.normalize(emb)
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        mean_l2r = mean_r2l = mrr_l2r = mrr_r2l = 0.0

        distance = pairwise_distances(emb[self.test_left], emb[self.test_right])
        distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        for idx in range(self.test_left.shape[0]):
            _, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mean_l2r += rank + 1
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1

        for idx in range(self.test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mean_r2l += rank + 1
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1

        mean_l2r /= self.test_left.size(0)
        mean_r2l /= self.test_right.size(0)
        mrr_l2r /= self.test_left.size(0)
        mrr_r2l /= self.test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)

        logger.info(
            f"{name} avg: acc@{top_k}={((acc_l2r + acc_r2l) / 2)}, mr={(mean_l2r + mean_r2l) / 2:.3f}, mrr={(mrr_l2r + mrr_r2l) / 2:.3f}"
        )

        return (
            (acc_l2r + acc_r2l) / 2,
            (mean_l2r + mean_r2l) / 2,
            (mrr_l2r + mrr_r2l) / 2,
        )


if __name__ == "__main__":
    args = HUMEA.parse_options(argparse.ArgumentParser())
    repo_root = Path(__file__).resolve().parent
    job_type = "ablation" if args.without else os.environ.get("HUMEA_JOB_TYPE", "main")
    name = os.environ.get("WANDB_NAME", f"{Path(args.file_dir).name}-{args.rate}-seed{args.seed}-without{args.without}")
    tracker = ExperimentTracker.start(
        config={**vars(args), "dataset": Path(args.file_dir).name,
                "experiment_id": os.environ.get("HUMEA_EXPERIMENT_ID", name)},
        repo_root=repo_root,
        output_dir=Path(os.environ.get("HUMEA_TRACKING_DIR", repo_root / "artifacts" / "tracking" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))),
        mode=os.environ.get("WANDB_MODE", "offline"),
        project=os.environ.get("WANDB_PROJECT", "humea-reproduction"),
        entity=os.environ.get("WANDB_ENTITY"), name=name, job_type=job_type,
    )
    exit_code = 1
    model = None
    try:
        # Seed/model initialization follows tracker startup so SDK initialization
        # cannot alter the random state used by HUMEA.
        model = HUMEA(args=args, tracker=tracker)
        model.train()
        if best_result is None:
            raise RuntimeError("Training ended without an evaluation; increase --epochs or reduce --check_point")
        (epoch, name, acc, mr, mrr) = best_result
        logger.info(
            f"Best avg epoch <{epoch}>: acc@{top_k}={acc}, mr={mr:.6f}, mrr={mrr:.6f}"
        )
        exit_code = 0
    finally:
        try:
            logger.complete()
            if model is not None:
                tracker.attach_file(model.training_log_path)
        except Exception:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish(exit_code=exit_code)
