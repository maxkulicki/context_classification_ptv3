"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import io
import os
import numpy as np
import wandb
from PIL import Image as PILImage
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        num_classes = self.trainer.cfg.data.num_classes
        confusion = torch.zeros(
            num_classes, num_classes, dtype=torch.long, device="cuda"
        )
        all_preds_list = []
        all_labels_list = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            # Accumulate confusion matrix on GPU
            indices = label * num_classes + pred
            confusion += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            all_preds_list.append(pred.cpu())
            all_labels_list.append(label.cpu())
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        # Sync confusion matrix across GPUs
        if comm.get_world_size() > 1:
            dist.all_reduce(confusion)
        cm = confusion.cpu().numpy()
        all_preds = torch.cat(all_preds_list).numpy()
        all_labels = torch.cat(all_labels_list).numpy()

        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )

        # Derive F1, precision, recall from confusion matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision_cls = tp / (tp + fp + 1e-10)
        recall_cls = tp / (tp + fn + 1e-10)
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-10)
        macro_f1 = np.mean(f1_cls)
        support = cm.sum(axis=1)
        weighted_f1 = np.average(f1_cls, weights=support)
        macro_precision = np.mean(precision_cls)
        macro_recall = np.mean(recall_cls)

        self.trainer.logger.info(
            "Val result: macro_F1/weighted_F1/macro_P/macro_R "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                macro_f1, weighted_f1, macro_precision, macro_recall
            )
        )
        for i in range(num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} F1/Precision/Recall: "
                "{f1:.4f}/{precision:.4f}/{recall:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    f1=f1_cls[i],
                    precision=precision_cls[i],
                    recall=recall_cls[i],
                )
            )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb_dict = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mIoU": m_iou,
                    "val/mAcc": m_acc,
                    "val/allAcc": all_acc,
                    "val/macro_f1": macro_f1,
                    "val/weighted_f1": weighted_f1,
                    "val/macro_precision": macro_precision,
                    "val/macro_recall": macro_recall,
                }
                class_names = self.trainer.cfg.data.names
                for i in range(num_classes):
                    name = class_names[i]
                    wandb_dict[f"val/f1_{name}"] = f1_cls[i]
                    wandb_dict[f"val/precision_{name}"] = precision_cls[i]
                    wandb_dict[f"val/recall_{name}"] = recall_cls[i]
                wandb.log(wandb_dict, step=wandb.run.step)
                # Log confusion matrix plot
                wandb.log(
                    {
                        "val/confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_labels.tolist(),
                            preds=all_preds.tolist(),
                            class_names=list(class_names),
                        )
                    },
                    step=wandb.run.step,
                )

        # Save confusion matrix to disk for cross-fold aggregation
        np.save(
            os.path.join(self.trainer.cfg.save_path, "confusion_matrix.npy"), cm
        )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver
        self.trainer.comm_info["current_mAcc_value"] = m_acc  # for secondary checkpoint

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class DualValClsEvaluator(HookBase):
    """
    Evaluator for snapshot_10class_dual_val.
    After each epoch runs separate evaluations on val_id (in-distribution)
    and val_ood (plot-level spatial holdout).

    Per-epoch outputs saved to {save_path}/plots/ and logged to wandb:
      - Normalised confusion matrix heatmap for each split
      - Per-dataset accuracy barplot for each split

    Checkpoint signals written to comm_info:
      current_metric_value      → val_id allAcc  → model_best.pth
      current_metric_name       → "allAcc_id"
      current_mAcc_value        → val_id mAcc    → model_best_mAcc.pth
      current_metric_value_ood  → val_ood allAcc → model_best_ood.pth
    """

    def before_train(self):
        if self.trainer.cfg.evaluate:
            self._build_val_ood_loader()
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            for tag in ("val_id", "val_ood"):
                wandb.define_metric(f"{tag}/*", step_metric="Epoch")

    def _build_val_ood_loader(self):
        from pointcept.datasets import build_dataset, collate_fn as pt_collate_fn
        cfg = self.trainer.cfg
        val_ood_cfg = dict(cfg.data.val)
        val_ood_cfg["split"] = "val_ood"
        val_ood_data = build_dataset(val_ood_cfg)
        if comm.get_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_ood_data, shuffle=False
            )
        else:
            sampler = None
        self.val_ood_loader = torch.utils.data.DataLoader(
            val_ood_data,
            batch_size=self.trainer.cfg.batch_size_val_per_gpu,
            shuffle=False,
            num_workers=self.trainer.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=sampler,
            collate_fn=pt_collate_fn,
        )

    def after_epoch(self):
        if not self.trainer.cfg.evaluate:
            return
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Dual Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        metrics_id  = self._eval_split(self.trainer.val_loader,  "val_id")
        metrics_ood = self._eval_split(self.val_ood_loader,      "val_ood")
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Dual Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"]     = metrics_id["all_acc"]
        self.trainer.comm_info["current_metric_name"]      = "allAcc_id"
        self.trainer.comm_info["current_mAcc_value"]       = metrics_id["m_acc"]
        self.trainer.comm_info["current_metric_value_ood"] = metrics_ood["all_acc"]

    def _eval_split(self, loader, split_name):
        num_classes = self.trainer.cfg.data.num_classes
        class_names = list(self.trainer.cfg.data.names)
        ignore_idx  = self.trainer.cfg.data.ignore_index

        source_names = getattr(loader.dataset, "source_names", [])
        n_sources    = len(source_names)

        confusion      = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")
        per_ds_correct = torch.zeros(n_sources, dtype=torch.long, device="cuda")
        per_ds_total   = torch.zeros(n_sources, dtype=torch.long, device="cuda")
        all_preds_list, all_labels_list = [], []
        total_loss, total_loss_main, n_batches = 0.0, 0.0, 0

        # Auxiliary head tracking (populated on first batch if aux logits exist)
        aux_keys = None          # list of aux branch names, e.g. ["lidar", "ctx_ae", "ctx_sinr"]
        aux_confusions = {}      # {name: (num_classes, num_classes) confusion tensor}
        aux_losses = {}          # {name: running sum of CE loss}

        for i, input_dict in enumerate(loader):
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            pred  = output_dict["cls_logits"].max(1)[1]
            label = input_dict["category"]
            total_loss += output_dict["loss"].item()
            if "loss_main" in output_dict:
                total_loss_main += output_dict["loss_main"].item()
            else:
                total_loss_main += output_dict["loss"].item()
            n_batches  += 1

            valid = (label != ignore_idx)
            indices = label[valid] * num_classes + pred[valid]
            confusion += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            all_preds_list.append(pred.cpu())
            all_labels_list.append(label.cpu())

            # Discover and accumulate auxiliary head outputs
            if aux_keys is None:
                aux_keys = []
                for okey in output_dict:
                    if okey.startswith("aux_logits_"):
                        name = okey[len("aux_logits_"):]
                        aux_keys.append(name)
                        aux_confusions[name] = torch.zeros(
                            num_classes, num_classes, dtype=torch.long, device="cuda")
                        aux_losses[name] = 0.0
            for name in aux_keys:
                aux_pred = output_dict[f"aux_logits_{name}"].max(1)[1]
                aux_idx = label[valid] * num_classes + aux_pred[valid]
                aux_confusions[name] += torch.bincount(
                    aux_idx, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
                aux_losses[name] += torch.nn.functional.cross_entropy(
                    output_dict[f"aux_logits_{name}"][valid],
                    label[valid], ignore_index=ignore_idx
                ).item()

            # Per-dataset tracking
            if n_sources > 0 and "source_id" in input_dict:
                sid = input_dict["source_id"]
                if sid.dim() == 2:
                    sid = sid.squeeze(1)
                ones = torch.ones(valid.sum(), dtype=torch.long, device="cuda")
                per_ds_total.index_add_(0, sid[valid], ones)
                per_ds_correct.index_add_(0, sid[valid], (pred == label)[valid].long())

            self.trainer.logger.info(
                f"{split_name} [{i+1}/{len(loader)}] Loss {output_dict['loss'].item():.4f}"
            )

        if comm.get_world_size() > 1:
            dist.all_reduce(confusion)
            if n_sources > 0:
                dist.all_reduce(per_ds_correct)
                dist.all_reduce(per_ds_total)
            for name in (aux_keys or []):
                dist.all_reduce(aux_confusions[name])

        # Compute aux head metrics (mAcc, allAcc, loss)
        aux_metrics = {}
        for name in (aux_keys or []):
            acm = aux_confusions[name].cpu().numpy()
            a_tp = np.diag(acm)
            a_acc_class = a_tp / (acm.sum(1) + 1e-10)
            aux_metrics[name] = dict(
                m_acc=float(np.mean(a_acc_class)),
                all_acc=float(a_tp.sum() / (acm.sum() + 1e-10)),
                loss=aux_losses[name] / max(n_batches, 1),
            )

        cm         = confusion.cpu().numpy()
        all_preds  = torch.cat(all_preds_list).numpy()
        all_labels = torch.cat(all_labels_list).numpy()

        tp  = np.diag(cm)
        fp  = cm.sum(0) - tp
        fn  = cm.sum(1) - tp
        acc_class     = tp / (cm.sum(1) + 1e-10)
        iou_class     = tp / (tp + fp + fn + 1e-10)
        precision_cls = tp / (tp + fp + 1e-10)
        recall_cls    = tp / (tp + fn + 1e-10)
        f1_cls        = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-10)
        m_acc       = float(np.mean(acc_class))
        m_iou       = float(np.mean(iou_class))
        all_acc     = float(np.diag(cm).sum() / (cm.sum() + 1e-10))
        macro_f1    = float(np.mean(f1_cls))
        weighted_f1 = float(np.average(f1_cls, weights=cm.sum(1)))
        loss_avg      = total_loss / max(n_batches, 1)
        loss_main_avg = total_loss_main / max(n_batches, 1)

        per_ds_acc = {}
        if n_sources > 0:
            c_np = per_ds_correct.cpu().numpy()
            t_np = per_ds_total.cpu().numpy()
            for s_idx, s_name in enumerate(source_names):
                if t_np[s_idx] > 0:
                    per_ds_acc[s_name] = float(c_np[s_idx] / t_np[s_idx])

        # Log to console
        self.trainer.logger.info(
            f"{split_name}: mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}"
        )
        for i in range(num_classes):
            self.trainer.logger.info(
                f"  {split_name} Class_{i}-{class_names[i]}: "
                f"iou/acc {iou_class[i]:.4f}/{acc_class[i]:.4f}"
            )
        for ds_name, acc in sorted(per_ds_acc.items()):
            self.trainer.logger.info(f"  {split_name} dataset {ds_name}: acc {acc:.4f}")
        for name, am in aux_metrics.items():
            self.trainer.logger.info(
                f"  {split_name} aux_{name}: loss {am['loss']:.4f} "
                f"mAcc {am['m_acc']:.4f} allAcc {am['all_acc']:.4f}"
            )

        # Save plots (main process only)
        epoch = self.trainer.epoch + 1
        if comm.is_main_process():
            plots_dir = os.path.join(self.trainer.cfg.save_path, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            cm_fig  = self._make_confusion_matrix(cm, class_names, split_name, epoch)
            cm_path = os.path.join(plots_dir, f"epoch{epoch:03d}_{split_name}_cm.png")
            cm_fig.savefig(cm_path, bbox_inches="tight", dpi=120)
            plt.close(cm_fig)

            bar_fig  = self._make_per_dataset_bar(per_ds_acc, split_name, epoch)
            bar_path = os.path.join(plots_dir, f"epoch{epoch:03d}_{split_name}_per_ds.png")
            bar_fig.savefig(bar_path, bbox_inches="tight", dpi=120)
            plt.close(bar_fig)

            np.save(
                os.path.join(self.trainer.cfg.save_path, f"confusion_matrix_{split_name}.npy"), cm
            )

            if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
                wandb_dict = {
                    "Epoch":                       epoch,
                    f"{split_name}/loss":          loss_main_avg,
                    f"{split_name}/loss_total":    loss_avg,
                    f"{split_name}/mIoU":          m_iou,
                    f"{split_name}/mAcc":          m_acc,
                    f"{split_name}/allAcc":        all_acc,
                    f"{split_name}/macro_f1":      macro_f1,
                    f"{split_name}/weighted_f1":   weighted_f1,
                }
                for i, name in enumerate(class_names):
                    wandb_dict[f"{split_name}/f1_{name}"]  = float(f1_cls[i])
                    wandb_dict[f"{split_name}/acc_{name}"] = float(acc_class[i])
                for ds_name, acc in per_ds_acc.items():
                    wandb_dict[f"{split_name}/ds_acc_{ds_name}"] = acc
                for name, am in aux_metrics.items():
                    wandb_dict[f"{split_name}/aux_{name}_loss"]   = am["loss"]
                    wandb_dict[f"{split_name}/aux_{name}_mAcc"]   = am["m_acc"]
                    wandb_dict[f"{split_name}/aux_{name}_allAcc"] = am["all_acc"]
                wandb.log(wandb_dict, step=wandb.run.step)
                wandb.log(
                    {f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=all_labels.tolist(),
                        preds=all_preds.tolist(), class_names=class_names,
                    )},
                    step=wandb.run.step,
                )
                wandb.log({
                    f"{split_name}/cm_plot":    wandb.Image(cm_path),
                    f"{split_name}/per_ds_bar": wandb.Image(bar_path),
                }, step=wandb.run.step)

        return dict(all_acc=all_acc, m_acc=m_acc, m_iou=m_iou, loss=loss_avg)

    def _make_confusion_matrix(self, cm, class_names, split_name, epoch):
        n       = len(class_names)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names, fontsize=8)
        for r in range(n):
            for c in range(n):
                ax.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if cm_norm[r, c] > 0.5 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{split_name} confusion — epoch {epoch}")
        fig.tight_layout()
        return fig

    def _make_per_dataset_bar(self, per_ds_acc, split_name, epoch):
        if not per_ds_acc:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No source_id data", ha="center", va="center", transform=ax.transAxes)
            return fig
        names = list(per_ds_acc.keys())
        accs  = [per_ds_acc[n] for n in names]
        order = np.argsort(accs)
        names = [names[i] for i in order]
        accs  = [accs[i]  for i in order]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.6)))
        bars = ax.barh(names, accs, color="steelblue")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{split_name} per-dataset accuracy — epoch {epoch}")
        for bar, acc in zip(bars, accs):
            ax.text(min(acc + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                    f"{acc:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        return fig

    def after_train(self):
        self.trainer.logger.info(
            "Best allAcc_id: {:.4f}".format(self.trainer.best_metric_value)
        )


def _log_normalized_confusion_matrix(cm, class_names, wandb_key, epoch):
    """Log a row-normalised confusion matrix as a wandb.Image heatmap."""
    num_classes = len(class_names)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(7, num_classes - 1)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(f"Normalised confusion matrix  (epoch {epoch})", fontsize=10)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if val > 0.55 else "black",
            )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    img = wandb.Image(PILImage.open(buf), caption=f"Epoch {epoch}")
    plt.close(fig)
    wandb.log({wandb_key: img, "Epoch": epoch}, step=wandb.run.step)


@HOOKS.register_module()
class UniversalClsEvaluator(HookBase):
    """Evaluator for UniversalCtxCls-v1m1 (Phase 2).

    Extends the standard ClsEvaluator with:
    - Per-source auxiliary head validation accuracy
      (logged as val/aux_acc_{source} for ae/topo/sinr/gpn)
    - Majority-class accuracy as a reference baseline
      (logged as val/majority_class_acc)
    - Row-normalised confusion matrix heatmap (wandb.Image)
      instead of raw counts only

    The model must return aux_logits_{source} tensors during validation.
    This is done automatically by UniversalCtxCls-v1m1 when category labels
    are present in the input dict.
    """

    # Source names in the same order as UniversalCtxCls.SOURCE_NAMES
    SOURCE_NAMES = ["ae", "topo", "sinr", "gpn"]

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        num_classes = self.trainer.cfg.data.num_classes

        # Main confusion matrix (for primary metrics)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")

        # Aux head accuracy accumulators: correct and total counts per source
        aux_correct = {s: torch.zeros(1, dtype=torch.long, device="cuda") for s in self.SOURCE_NAMES}
        aux_total = torch.zeros(1, dtype=torch.long, device="cuda")

        all_preds_list = []
        all_labels_list = []

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            cls_logits = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = cls_logits.max(1)[1]
            label = input_dict["category"]

            # Valid (non-ignored) mask
            valid = label != self.trainer.cfg.data.ignore_index

            # Main metrics
            intersection, union, target = intersection_and_union_gpu(
                pred, label, num_classes, self.trainer.cfg.data.ignore_index,
            )
            indices = label[valid] * num_classes + pred[valid]
            confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )
            all_preds_list.append(pred.cpu())
            all_labels_list.append(label.cpu())

            # Aux head accuracy
            for src in self.SOURCE_NAMES:
                aux_key = f"aux_logits_{src}"
                if aux_key in output_dict:
                    aux_pred = output_dict[aux_key].max(1)[1]
                    aux_correct[src] += (aux_pred[valid] == label[valid]).sum()
            aux_total += valid.sum()

            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = (
                intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(),
            )
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] Loss {loss:.4f}".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        # Sync confusion matrix and aux counters across GPUs
        if comm.get_world_size() > 1:
            dist.all_reduce(confusion)
            for src in self.SOURCE_NAMES:
                dist.all_reduce(aux_correct[src])
            dist.all_reduce(aux_total)

        cm = confusion.cpu().numpy()
        all_preds = torch.cat(all_preds_list).numpy()
        all_labels = torch.cat(all_labels_list).numpy()
        total_valid = aux_total.item()

        # ── Primary metrics ────────────────────────────────────────────────
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(m_iou, m_acc, all_acc)
        )
        for i in range(num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i, name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i], accuracy=acc_class[i],
                )
            )

        # ── F1 / precision / recall ────────────────────────────────────────
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision_cls = tp / (tp + fp + 1e-10)
        recall_cls = tp / (tp + fn + 1e-10)
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-10)
        macro_f1 = np.mean(f1_cls)
        support = cm.sum(axis=1)
        weighted_f1 = np.average(f1_cls, weights=support)
        macro_precision = np.mean(precision_cls)
        macro_recall = np.mean(recall_cls)

        self.trainer.logger.info(
            "Val result: macro_F1/weighted_F1/macro_P/macro_R "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(macro_f1, weighted_f1, macro_precision, macro_recall)
        )
        for i in range(num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} F1/Precision/Recall: {f1:.4f}/{precision:.4f}/{recall:.4f}".format(
                    idx=i, name=self.trainer.cfg.data.names[i],
                    f1=f1_cls[i], precision=precision_cls[i], recall=recall_cls[i],
                )
            )

        # ── Auxiliary head accuracy ────────────────────────────────────────
        # Majority-class baseline: most frequent class in val labels
        majority_class_acc = cm.sum(axis=1).max() / (cm.sum() + 1e-10)
        aux_acc = {
            src: aux_correct[src].item() / (total_valid + 1e-10)
            for src in self.SOURCE_NAMES
        }
        self.trainer.logger.info(
            "Aux head accuracy (majority baseline {:.3f}): "
            "AE={ae:.3f}  Topo={topo:.3f}  SINR={sinr:.3f}  GPN={gpn:.3f}".format(
                majority_class_acc,
                ae=aux_acc["ae"], topo=aux_acc["topo"],
                sinr=aux_acc["sinr"], gpn=aux_acc["gpn"],
            )
        )

        # ── W&B logging ────────────────────────────────────────────────────
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)

            if self.trainer.cfg.enable_wandb:
                class_names = self.trainer.cfg.data.names
                wandb_dict = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mIoU": m_iou,
                    "val/mAcc": m_acc,
                    "val/allAcc": all_acc,
                    "val/macro_f1": macro_f1,
                    "val/weighted_f1": weighted_f1,
                    "val/macro_precision": macro_precision,
                    "val/macro_recall": macro_recall,
                    # Aux head accuracy
                    "val/majority_class_acc": majority_class_acc,
                    **{f"val/aux_acc_{src}": v for src, v in aux_acc.items()},
                }
                for i in range(num_classes):
                    name = class_names[i]
                    wandb_dict[f"val/f1_{name}"] = f1_cls[i]
                    wandb_dict[f"val/precision_{name}"] = precision_cls[i]
                    wandb_dict[f"val/recall_{name}"] = recall_cls[i]
                wandb.log(wandb_dict, step=wandb.run.step)

                # Normalised confusion matrix heatmap
                _log_normalized_confusion_matrix(
                    cm, list(class_names), "val/confusion_matrix_normalized", current_epoch
                )
                # Also keep the interactive raw-count matrix
                wandb.log(
                    {
                        "val/confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_labels.tolist(),
                            preds=all_preds.tolist(),
                            class_names=list(class_names),
                        )
                    },
                    step=wandb.run.step,
                )

        # Save confusion matrix to disk for offline analysis
        np.save(os.path.join(self.trainer.cfg.save_path, "confusion_matrix.npy"), cm)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc
        self.trainer.comm_info["current_metric_name"] = "allAcc"
        self.trainer.comm_info["current_mAcc_value"] = m_acc

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = {}
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            data_name = input_dict.pop("name")[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = {}
            for _ in range(len(scenes_sync)):
                r = scenes_sync.pop()
                scenes.update(r)
                del r
            scenes = list(scenes.values())
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            self.trainer.logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
                self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
                self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "val/loss": loss_avg,
                            "val/mAP": all_ap,
                            "val/AP50": all_ap_50,
                            "val/AP25": all_ap_25,
                        },
                        step=wandb.run.step,
                    )
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
            self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
            self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
