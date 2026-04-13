/**
 * 3D embroidery conversion flow composable: 2-step state machine.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';

export function useThreed(form, currentTab, fetchJobs, fetchCredit, userValue) {
    const tdFlowState = ref('idle');
    const tdStep1Data = ref([]);
    const tdSettings = ref({ skip_step1_review: false, one_shot: false });

    let _setSubmitting = null;
    const linkSubmitting = (setFn) => { _setSubmitting = setFn; };
    const setSubmitting = (v) => { if (_setSubmitting) _setSubmitting(v); };

    const cancelThreedFlow = () => {
        tdFlowState.value = 'idle';
        setSubmitting(false);
    };

    const runThreedJobSubmit = async () => {
        tdFlowState.value = 'td_job_submitting';
        try {
            const tdData = tdStep1Data.value
                .filter(item => item.description && item.description.trim())
                .map(item => ({ description: item.description.trim(), image_name: item.image_name }));
            if (!tdData.length) throw new Error('没有有效的图案描述，请手动输入后重试');

            const fd = new FormData();
            fd.append('mode', 'threed');
            fd.append('model_id', form.value.model_id);
            fd.append('target_ratio', form.value.target_ratio || '1:1');
            fd.append('batch_size', 1);
            fd.append('threed_data', JSON.stringify(tdData));
            fd.append('prompts', '');
            fd.append('negative_prompt', form.value.negative_prompt || '');
            fd.append('template_name', '');

            const res = await authFetch('/api/jobs', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.id) throw new Error(data.error || '任务提交失败');

            tdFlowState.value = 'idle';
            setSubmitting(false);
            form.value.files = [];
            currentTab.value = 'jobs';
            fetchJobs(userValue());
            fetchCredit();
        } catch (e) {
            alert('任务提交失败：' + e.message);
            cancelThreedFlow();
        }
    };

    const proceedToThreedSubmit = async () => {
        setSubmitting(true);
        await runThreedJobSubmit();
    };

    const startThreedFlow = async () => {
        if (form.value.files.length === 0) { alert('请上传至少一张图片'); return; }
        setSubmitting(true);
        tdFlowState.value = 'td_step1_loading';
        try {
            const fd = new FormData();
            form.value.files.forEach(f => fd.append('images', f));
            const res = await authFetch('/api/threed/understand', { method: 'POST', body: fd });
            if (!res.ok) { const d = await res.json(); throw new Error(d.error || 'AI 图案分析失败'); }
            const data = await res.json();
            tdStep1Data.value = data.items;
            if (tdSettings.value.one_shot || tdSettings.value.skip_step1_review) {
                await runThreedJobSubmit();
            } else {
                tdFlowState.value = 'td_step1_review';
                setSubmitting(false);
            }
        } catch (e) {
            alert('图案分析失败：' + e.message);
            cancelThreedFlow();
        }
    };

    return {
        tdFlowState, tdStep1Data, tdSettings,
        cancelThreedFlow, proceedToThreedSubmit, startThreedFlow, linkSubmitting,
    };
}
