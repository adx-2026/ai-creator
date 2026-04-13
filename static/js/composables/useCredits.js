/**
 * Credit estimation and fetching composable.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';
import { CREDITS_PER_YUAN, IMAGE_PRICE, VIDEO_PRICE, VL_PRICE, VL_EST_INPUT, VL_EST_OUTPUT } from '../constants.js';

export function useCredits() {
    const userCredit = ref(null);

    const isAliyunModel = (id) => id && (id.startsWith('qwen') || id.startsWith('wan'));

    const fetchCredit = async () => {
        try {
            const res = await authFetch('/api/credit');
            if (res.ok) {
                const d = await res.json();
                userCredit.value = d.credit;
            }
        } catch (e) { }
    };

    const estimateCredits = (form, calculateTotalTasks, ecSettings, multiLineCount) => {
        const modelId = form.model_id;
        if (!isAliyunModel(modelId)) return 0;
        const mode = form.mode;

        if (mode === 'video') {
            const duration = form.videoParams.duration || 5;
            return Math.round((VIDEO_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN * duration * 10000) / 10000;
        }
        if (mode === 'extract') {
            const imgCount = form.files.length || 1;
            const batchSize = form.batch_size || 1;
            const vlCost = (VL_EST_INPUT / 1e6 * VL_PRICE.input + VL_EST_OUTPUT / 1e6 * VL_PRICE.output) * CREDITS_PER_YUAN;
            const imgCost = (IMAGE_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN;
            return Math.round(imgCount * (vlCost + batchSize * imgCost) * 10000) / 10000;
        }
        if (mode === 'multi_t2i') {
            return Math.round(calculateTotalTasks() * (IMAGE_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN * 10000) / 10000;
        }
        if (mode === 'multi_video') {
            const duration = form.videoParams.duration || 5;
            return Math.round(calculateTotalTasks() * (VIDEO_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN * duration * 10000) / 10000;
        }
        if (mode === 'ecommerce') {
            const imgCount = form.files.length || 1;
            const sceneCount = ecSettings.scene_count || 3;
            const vlCost = (VL_EST_INPUT / 1e6 * VL_PRICE.input + VL_EST_OUTPUT / 1e6 * VL_PRICE.output) * CREDITS_PER_YUAN;
            const textCost = (400 / 1e6 * 1.0 + 400 / 1e6 * 8.0) * CREDITS_PER_YUAN;
            const imgCost = (IMAGE_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN;
            return Math.round(imgCount * (vlCost + textCost + sceneCount * imgCost) * 10000) / 10000;
        }
        if (mode === 'threed') {
            const imgCount = form.files.length || 1;
            const vlCost = (VL_EST_INPUT / 1e6 * VL_PRICE.input + VL_EST_OUTPUT / 1e6 * VL_PRICE.output) * CREDITS_PER_YUAN;
            const imgCost = (IMAGE_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN;
            return Math.round(imgCount * (vlCost + imgCost) * 10000) / 10000;
        }
        return Math.round(calculateTotalTasks() * (IMAGE_PRICE[modelId] ?? 0.5) * CREDITS_PER_YUAN * 10000) / 10000;
    };

    return { userCredit, isAliyunModel, fetchCredit, estimateCredits };
}
