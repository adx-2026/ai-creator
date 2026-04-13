/**
 * Application constants: mode labels, ratio options, pricing tables.
 */

export const MODE_LABELS = {
    t2i: '文本生图', i2i: '图像生图', fission: '图片裂变',
    convert: 'AI尺寸重绘', video: '视频生成', multi_t2i: '多图生成',
    multi_video: '多视频生成', extract: '图案提取生图',
    ecommerce: '电商场景生图', threed: '3D图片转换',
};

export const RATIO_OPTIONS = [
    { value: '1:1', label: '1:1 正方形' },
    { value: '3:2', label: '3:2 横向/相机' },
    { value: '2:3', label: '2:3 竖向/相机' },
    { value: '4:3', label: '4:3 横向/显示器' },
    { value: '3:4', label: '3:4 竖向' },
    { value: '16:9', label: '16:9 宽屏' },
    { value: '9:16', label: '9:16 手机全屏' },
];

// Credit estimation (mirrors backend pricing)
export const CREDITS_PER_YUAN = 10;

export const IMAGE_PRICE = {
    'qwen-image-2.0-pro': 0.5, 'qwen-image-2.0-pro-2026-03-03': 0.5,
    'qwen-image-2.0': 0.2, 'qwen-image-2.0-2026-03-03': 0.2,
    'qwen-image-max': 0.5, 'qwen-image-max-2025-12-30': 0.5,
    'qwen-image-plus': 0.2, 'qwen-image-plus-2026-01-09': 0.2,
    'qwen-image': 0.25, 'qwen-image-edit-max': 0.5,
    'qwen-image-edit-max-2026-01-16': 0.5, 'qwen-image-edit-plus': 0.2,
    'qwen-image-edit': 0.3,
};

export const VIDEO_PRICE = {
    'wan2.6-t2v': 0.6, 'wan2.5-t2v-preview': 0.6,
    'wan2.2-t2v-plus': 0.14, 'wanx2.1-t2v-turbo': 0.24,
    'wanx2.1-t2v-plus': 0.70, 'wan2.6-i2v-flash': 0.3,
    'wan2.6-i2v-plus': 0.6, 'wan2.6-i2v-turbo': 0.3,
    'wanx2.1-i2v-turbo': 0.24, 'wanx2.1-i2v-plus': 0.70,
};

export const VL_PRICE = { input: 1.0, output: 10.0 };
export const VL_EST_INPUT = 3000;
export const VL_EST_OUTPUT = 500;
