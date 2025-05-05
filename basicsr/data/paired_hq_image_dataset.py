import cv2
import random
import torch
import os
from torch.utils import data as data
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from torchvision.io import encode_jpeg, decode_jpeg

# Assuming these imports exist and work as in the original context
from basicsr.data.data_util import paths_from_folder, paths_from_lmdb # Modified function names assumed
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedHQImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Reads GT image and dynamically generates LQ image by applying downscaling (2x)
    and 0 to 3 rounds of JPEG compression with random quality in [60, 90].
    The order of downscaling and all compression steps is fully randomized.

    There are three modes for finding GT images:
    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb. GT paths stored in lmdb.
    2. **meta_info_file**: Use meta information file to generate paths.
       If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        meta_info_file (str): Path for meta information file (for GT images).
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename (for GT images). Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        # scale (int): Implicitly set to 2 due to 2x downscaling for LQ generation.
        phase (str): 'train' or 'val'.
        color (str): Color space ('y' for Y channel only, 'rgb' otherwise). Default: 'rgb'.
        jpeg_range (list): Range for JPEG quality [min, max]. Default: [60, 90].
        jpeg_compress_num (list): Range for number of JPEG compressions [min, max]. Default: [0, 3].
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
    """

    def __init__(self, opt):
        super(PairedHQImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.scale = 2 # Hardcoded scale factor for LQ generation

        self.gt_folder = opt['dataroot_gt']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # Configure JPEG degradation parameters
        self.jpeg_range = opt.get('jpeg_range', [60, 90])
        self.jpeg_compress_num_range = opt.get('jpeg_compress_num', [0, 3])

        # Load GT paths
        print(self.filename_tmpl)
        for _, _, fl in os.walk(os.path.join(self.gt_folder)):
            self.paths = [os.path.join(self.gt_folder, f) for f in fl]

        # Ensure paths format is a list of dictionaries [{'gt_path': path1}, ...]
        if self.paths and isinstance(self.paths[0], str):
            self.paths = [{'gt_path': p} for p in self.paths]


    def _apply_jpeg_compression(self, img_tensor, quality):
        """Applies JPEG compression to a PyTorch tensor.

        Args:
            img_tensor (Tensor): Input image tensor (CHW, RGB, float32 [0, 1]).
            quality (int): JPEG quality factor (0-100).

        Returns:
            Tensor: Compressed and decompressed image tensor (CHW, RGB, float32 [0, 1]).
        """
        # Convert to uint8 [0, 255] for encode_jpeg
        img_tensor_uint8 = (img_tensor * 255.0).round().clamp(0, 255).to(torch.uint8)
        # Encode to JPEG bytes
        jpeg_bytes = encode_jpeg(img_tensor_uint8, quality=quality)
        # Decode back to tensor (uint8)
        decoded_tensor_uint8 = decode_jpeg(jpeg_bytes)
        # Convert back to float32 [0, 1]
        decoded_tensor_float32 = decoded_tensor_uint8.to(torch.float32) / 255.0
        return decoded_tensor_float32

    def _generate_lq(self, img_gt):
        """Generates LQ image from GT image with randomized degradation order."""

        # 0. Convert GT (NumPy HWC BGR float32) to Tensor (CHW RGB float32) for processing
        img_gt_tensor = img2tensor(img_gt, bgr2rgb=True, float32=True)

        # 1. Determine degradation parameters
        num_compressions = random.randint(self.jpeg_compress_num_range[0], self.jpeg_compress_num_range[1])
        jpeg_qualities = [random.randint(self.jpeg_range[0], self.jpeg_range[1]) for _ in range(num_compressions)]

        # 2. Define the downscale operation function
        def apply_downscale(tensor):
            h, w = tensor.shape[-2:]
            # Ensure target size is at least 1x1
            target_h = max(1, h // self.scale)
            target_w = max(1, w // self.scale)
            # Use BICUBIC interpolation for downscaling
            return resize(tensor, [target_h, target_w], interpolation=InterpolationMode.BICUBIC, antialias=True)

        # 3. Build the list of operations
        operations = []
        # Add the downscale operation (represented by a tuple marker)
        operations.append(('downscale', None))
        # Add the JPEG compression operations (represented by tuple markers)
        for quality in jpeg_qualities:
            operations.append(('jpeg', quality))

        # 4. Shuffle the order of operations
        random.shuffle(operations)

        # 5. Apply operations in the shuffled sequence
        current_img_tensor = img_gt_tensor.clone() # Work on a copy
        for op_type, op_param in operations:
            if op_type == 'downscale':
                current_img_tensor = apply_downscale(current_img_tensor)
            elif op_type == 'jpeg':
                quality = op_param
                current_img_tensor = self._apply_jpeg_compression(current_img_tensor, quality)
            # else: # Optional: handle unexpected operation types if necessary
            #     print(f"Warning: Unknown operation type '{op_type}' encountered.")

        # 6. Convert the final LQ tensor back to NumPy (HWC BGR float32)
        img_lq = current_img_tensor.permute(1, 2, 0).contiguous().numpy() # CHW -> HWC
        # Add .contiguous() before .numpy() if permute makes tensor non-contiguous
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR) # RGB -> BGR
        img_lq = img_lq.astype('float32')
        img_lq = img_lq.clip(0, 1) # Ensure range [0, 1]

        return img_lq


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt image. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True) # Reads as BGR HWC float32 [0,1]

        # Generate lq image dynamically
        img_lq = self._generate_lq(img_gt)

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Random crop needs GT and LQ numpy arrays (HWC)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)
            # Flip, rotation expects list of numpy arrays (HWC)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # Color space transform (optional)
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # Crop the unmatched GT images during validation or testing
        if self.opt['phase'] != 'train':
            h_lq, w_lq = img_lq.shape[0:2]
            target_h_gt, target_w_gt = h_lq * self.scale, w_lq * self.scale
            img_gt = img_gt[0:target_h_gt, 0:target_w_gt, :]
            # Add a check in case GT is somehow smaller than expected
            if img_gt.shape[0] != target_h_gt or img_gt.shape[1] != target_w_gt:
                 print(f"Warning: GT shape mismatch after crop in validation. GT: {img_gt.shape}, Target: ({target_h_gt}, {target_w_gt}), LQ: {img_lq.shape[:2]}")


        # BGR to RGB, HWC to CHW, numpy to tensor
        is_y_channel = 'color' in self.opt and self.opt['color'] == 'y'
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=not is_y_channel, float32=True)

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        lq_path = gt_path

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)