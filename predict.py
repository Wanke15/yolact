from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg
import torch

import os
from pathlib import Path

import matplotlib.pyplot as plt
import cv2


class MattingService:
    def __init__(self, model_path="./weights/yolact_im700_54_800000.pth", use_cuda=False):
        print('Loading model...', end='')
        self.use_cuda = use_cuda
        self.trained_model = model_path
        self.net = Yolact()
        self.net.load_weights(self.trained_model)
        self.net.eval()

        if self.use_cuda:
            self.net = self.net.cuda()

        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        print(' Done.')

    def process(self, image, top_k=1, score_threshold=0.6):
        # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
        with torch.no_grad():
            if image is not None:
                if ':' in image:
                    inp, _image_name = image.split(':')
                    self._infer_image(self.net, inp, _image_name, top_k, score_threshold)
                else:
                    _image_name = image.split('/')[-1].split('.')[0] + '.png'
                    out = os.path.join('results/', _image_name)
                    self._infer_image(self.net, image, out, top_k, score_threshold)
                return _image_name

    def _infer_image(self, net: Yolact, path, save_path, top_k, score_threshold):
        if self.use_cuda:
            frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        else:
            frame = torch.from_numpy(cv2.imread(path)).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)

        img_numpy = self.post_process(preds, frame, None, None, top_k, score_threshold, undo_transform=False)

        if save_path is None:
            img_numpy = img_numpy[:, :, (2, 1, 0, 3)]

        if save_path is None:
            plt.subplot()
            plt.imshow(img_numpy)
            plt.title(path)
            plt.show()
        else:
            # plt.subplot()
            # plt.imshow(img_numpy)
            # plt.title(path)
            # plt.show()
            cv2.imwrite(save_path, img_numpy)

    @staticmethod
    def post_process(dets_out, img, h, w, top_k=1, score_threshold=0.6, undo_transform=True):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb=False,
                            crop_masks=False,
                            score_threshold=score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:top_k]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        # After this, mask is of size [num_dets, h, w, 1]
        final_res = (img_gpu * 255).byte().cpu().numpy()
        final_res = cv2.cvtColor(final_res, cv2.COLOR_RGB2RGBA)

        if num_dets_to_consider == 0:
            return final_res

        masks = masks[:num_dets_to_consider, :, :, None]

        _mask = (masks * 255).byte().cpu().numpy()[0]

        # Then assign the mask to the last channel of the image
        final_res[:, :, 3] = _mask.squeeze()

        return final_res


if __name__ == '__main__':
    service = MattingService()
    image = './images/cat.jpg'
    service.process(image, 1, 0.09)
