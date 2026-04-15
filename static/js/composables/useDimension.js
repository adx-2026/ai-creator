/**
 * Product dimension annotation pipeline composable.
 * Single-step flow: upload image + dimension params → annotated image result.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';

export function useDimension(form, fetchCredit) {
    const dimState = ref('idle'); // idle | loading | done | error
    const dimResult = ref(null);
    const dimError = ref('');
    const dimSettings = ref({
        shape: 'rectangle',
        length: '',
        width: '',
        height: '',
        diameter: '',
        output_width: 1024,
        output_height: 1024,
        edit_model: 'qwen-image-edit-plus',
    });

    let _setSubmitting = null;
    const linkSubmitting = (setFn) => { _setSubmitting = setFn; };
    const setSubmitting = (v) => { if (_setSubmitting) _setSubmitting(v); };

    const cancelDimensionFlow = () => {
        dimState.value = 'idle';
        dimResult.value = null;
        dimError.value = '';
        setSubmitting(false);
    };

    const startDimensionFlow = async () => {
        if (form.value.files.length === 0) { alert('请上传产品图片'); return; }
        setSubmitting(true);
        dimState.value = 'loading';
        dimError.value = '';
        dimResult.value = null;

        try {
            const s = dimSettings.value;
            const fd = new FormData();
            fd.append('image', form.value.files[0]);
            fd.append('shape', s.shape);
            fd.append('edit_model', s.edit_model);
            fd.append('output_width', s.output_width);
            fd.append('output_height', s.output_height);

            if (s.shape === 'circle') {
                if (s.diameter) fd.append('diameter', s.diameter);
            } else {
                if (s.length) fd.append('length', s.length);
                if (s.width) fd.append('width', s.width);
                if (s.height) fd.append('height', s.height);
            }

            const res = await authFetch('/api/dimension/annotate', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || '尺寸图生成失败');

            dimResult.value = data;
            dimState.value = 'done';
            fetchCredit();
        } catch (e) {
            dimError.value = e.message;
            dimState.value = 'error';
            alert('尺寸图生成失败：' + e.message);
        } finally {
            setSubmitting(false);
        }
    };

    return {
        dimState, dimResult, dimError, dimSettings,
        cancelDimensionFlow, startDimensionFlow, linkSubmitting,
    };
}
