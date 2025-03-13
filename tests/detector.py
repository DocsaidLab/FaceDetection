from src.detector import Detector


def main():
    cfg = {
        'h': 640,
        'w': 640,
        'backbone_dict': {
            'name': 'scrfd_resv1e_10g',
            'in_channels': 3,
            'out_indices': [2, 3, 4],
        },
        'neck_dict': {
            'name': 'mm_pafpn',  # if use atorch.build_neck you need this
            'in_channels': [88, 88, 224],
            'out_channels': 56,
            'num_outs': 3,
            'add_extra_convs': 'on_output',
            'norm_cfg': {'type': 'BN'},
        },
        'head_dict': {
            'name': 'scrfd',
            'head': {
                'n_stack': 2,
                'n_levels': 3,
                'in_channels': 56,
                'hid_channels': 80,
                'norm': {
                    'name': 'GroupNorm',
                    'num_groups': 16,
                    'num_channels': 80,
                },
                'act': {
                    'name': 'ReLU',
                    'inplace': True,
                },
                'feat_share': True,
                'use_scale': True,
            },
            'prior': {
                'scales': [1.0, 2.0],
                'ratios': [1.0],
                'base_sizes': [16, 64, 256],
                'strides': [8, 16, 32],
            },
            'loss_weight': {
                'loc': 2.,
                'obj': 1.,
                'lmk5pt': 0.1,
            },
            'nms_cfg': {
                'score_th': 0.2,
                'nms_topk': 5000,
                'nms_th': 0.45,
            }
        }
    }
    detector = Detector(**cfg)
    print(detector)


main()
