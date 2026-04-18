/**
 * Product size annotation flow composable.
 * Single-step: user sets params → API call → result image.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';

export function useSizeAnnotation(form, fetchCredit, userValue) {
    const saFlowState = ref('idle'); // idle | loading | done
    const saResult = ref(null);     // { display_url, image_path, credit_cost }
    const saParams = ref({
        shape: 'square',
        length: '',
        width: '',
        height: '',
        diameter: '',
        output_width: '800',
        output_height: '800',
        edit_model: 'qwen-image-edit-plus',
    });

    let _setSubmitting = null;
    const linkSubmitting = (setFn) => { _setSubmitting = setFn; };
    const setSubmitting = (v) => { if (_setSubmitting) _setSubmitting(v); };

    const cancelSizeAnnotFlow = () => {
        saFlowState.value = 'idle';
        saResult.value = null;
        setSubmitting(false);
    };

    const resetSizeAnnotResult = () => {
        saResult.value = null;
        saFlowState.value = 'idle';
    };

    const startSizeAnnotFlow = async () => {
        if (form.value.files.length === 0) {
            alert('请上传一张产品图片');
            return;
        }

        setSubmitting(true);
        saFlowState.value = 'loading';
        saResult.value = null;

        try {
            const p = saParams.value;
            const outputSize = `${p.output_width || 800}*${p.output_height || 800}`;

            const fd = new FormData();
            fd.append('image', form.value.files[0]);
            fd.append('shape', p.shape);
            fd.append('length', p.length);
            fd.append('width', p.width);
            fd.append('height', p.height);
            fd.append('diameter', p.diameter);
            fd.append('output_size', outputSize);
            fd.append('model_id', p.edit_model);

            const res = await authFetch('/api/size-annotation/generate', {
                method: 'POST',
                body: fd,
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || '生成失败');

            saResult.value = data;
            saFlowState.value = 'done';
            fetchCredit();
        } catch (e) {
            alert('尺寸标注图生成失败：' + e.message);
            saFlowState.value = 'idle';
        } finally {
            setSubmitting(false);
        }
    };

    return {
        saFlowState, saResult, saParams,
        cancelSizeAnnotFlow, resetSizeAnnotResult,
        startSizeAnnotFlow, linkSubmitting,
    };
}
