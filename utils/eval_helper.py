import glob
import logging
import os

import numpy as np
import tabulate
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn import metrics


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


from typing import Tuple, Dict, Optional, Sequence

class SaveAnomap:

    class Empty:

        def wait(self):
            ...

    def __init__(self, save_path: str, image_size: Tuple[int, int]) -> None:
        
        from collections.abc import Sequence as SequenceAbc

        self.save_path = save_path
        self.image_size = image_size
        self.need_pad = False
        self.minimum = float('inf')
        self.anomap: Dict[str, Tuple[np.ndarray, float, Optional[str]]] = {}

        assert (isinstance(image_size, SequenceAbc) and len(image_size) == 2 and
                all(isinstance(val, int) for val in image_size))

    def record_batch(self, anomaly_maps: Tensor, pred_scores: Tensor,
                     image_path: Sequence[str], mask_path: Sequence[str]) -> None:
        import torch
        from collections.abc import Sequence as SequenceAbc

        assert isinstance(image_path, SequenceAbc) and all(isinstance(val, str) for val in image_path)
        assert isinstance(mask_path, SequenceAbc) and all(isinstance(val, str) for val in mask_path)
        assert len(mask_path) == len(image_path)
        assert isinstance(anomaly_maps, Tensor) and anomaly_maps.shape[0] == len(image_path)
        anomaly_maps = anomaly_maps.squeeze()
        if len(image_path) == 1:
            anomaly_maps = anomaly_maps.unsqueeze_(0)
        assert anomaly_maps.ndim == 3, str(anomaly_maps.shape)
        assert isinstance(pred_scores, torch.Tensor) and pred_scores.numel() == len(image_path)
        pred_scores = pred_scores.view(-1)

        assert self.image_size[0] >= anomaly_maps.shape[1]
        assert self.image_size[1] >= anomaly_maps.shape[2]
        if anomaly_maps.shape[1] < self.image_size[0] or anomaly_maps.shape[2] < self.image_size[1]:
            self.need_pad = True
            self.minimum = min(anomaly_maps.min(), self.minimum)

        for path, mask, anomap, score in zip(image_path, mask_path, anomaly_maps, pred_scores):
            self.anomap[path] = (anomap.cpu().numpy(), float(score), mask if mask != '' else None)

    def collect(self) -> None:

        import torch.distributed as dist
        import pickle

        if self.need_pad:
            def pad(anomap: np.ndarray) -> np.ndarray:
                y = (self.image_size[0] - anomap.shape[0]) // 2
                x = (self.image_size[1] - anomap.shape[1]) // 2
                newmap = np.full(self.image_size, self.minimum, dtype=anomap.dtype)
                newmap[y: y + anomap.shape[0], x: x + anomap.shape[1]] = anomap
                return newmap

            self.anomap = {key: (pad(anomap), score, mask)
                           for key, (anomap, score, mask) in self.anomap.items()}

        assert dist.is_initialized()

        if dist.get_backend() == 'nccl':
            pg = dist.new_group(list(range(dist.get_world_size())),
                                backend=dist.Backend.GLOO)
        else:
            pg = dist.GroupMember.WORLD

        my_rank = dist.get_rank()
        for rank in range(1, dist.get_world_size()):
            if my_rank == rank:
                buf = torch.from_numpy(np.frombuffer(pickle.dumps(self.anomap), dtype=np.uint8))
                cnt = torch.tensor(buf.numel(), dtype=torch.int64)
                req = dist.isend(cnt, dst=0, group=pg)

            elif my_rank == 0:
                buf = torch.empty((0,), dtype=torch.uint8)
                cnt = torch.empty((), dtype=torch.int64)
                req = dist.irecv(cnt, src=rank, group=pg)

            else:
                buf = torch.empty((0,), dtype=torch.uint8)
                cnt = torch.empty((), dtype=torch.int64)
                req = self.Empty()

            req.wait()

            if my_rank == rank:
                req = dist.isend(buf, dst=0, group=pg)

            elif my_rank == 0:
                buf = torch.empty((cnt.item(),), dtype=torch.uint8)
                req = dist.irecv(buf, src=rank, group=pg)

            else:
                req = self.Empty()

            req.wait()

            if my_rank == 0:
                self.anomap.update(pickle.loads(
                    np.ndarray.tobytes(buf.numpy())
                ))

            del buf, cnt
            dist.barrier(pg)

        if my_rank == 0:
            with open(self.save_path, 'wb') as fp:
                pickle.dump(self.anomap, fp)
        dist.barrier(pg)

        if pg is not dist.GroupMember.WORLD:
            dist.destroy_process_group(pg)


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int32)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}


def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
