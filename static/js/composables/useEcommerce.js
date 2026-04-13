/**
 * E-commerce scene generation flow composable: 3-step state machine.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';

export function useEcommerce(form, currentTab, fetchJobs, fetchCredit, userValue) {
    const ecFlowState = ref('idle');
    const ecStep1Data = ref([]);
    const ecStep2Data = ref([]);
    const ecSettings = ref({ skip_step1_review: false, skip_step2_review: false, one_shot: false, scene_count: 3 });

    const isSubmitting = { value: false }; // will be linked externally

    let _setSubmitting = null;
    const linkSubmitting = (setFn) => { _setSubmitting = setFn; };
    const setSubmitting = (v) => { if (_setSubmitting) _setSubmitting(v); };

    const cancelEcommerceFlow = () => {
        ecFlowState.value = 'idle';
        setSubmitting(false);
    };

    const runEcommerceJobSubmit = async () => {
        ecFlowState.value = 'job_submitting';
        try {
            const ecData = ecStep2Data.value
                .map(item => ({ image_path: item.image_path, scenes: (item.scenes || []).filter(s => s.trim()) }))
                .filter(item => item.scenes.length > 0);
            if (!ecData.length) throw new Error('没有有效的场景数据，请至少保留一个场景');

            const fd = new FormData();
            fd.append('mode', 'ecommerce');
            fd.append('model_id', form.value.model_id);
            fd.append('target_ratio', form.value.target_ratio || '1:1');
            fd.append('batch_size', 1);
            fd.append('ecommerce_data', JSON.stringify(ecData));
            fd.append('prompts', '');
            fd.append('negative_prompt', form.value.negative_prompt || '');
            fd.append('template_name', '');

            const res = await authFetch('/api/jobs', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.id) throw new Error(data.error || '任务提交失败');

            ecFlowState.value = 'idle';
            setSubmitting(false);
            form.value.files = [];
            currentTab.value = 'jobs';
            fetchJobs(userValue());
            fetchCredit();
        } catch (e) {
            alert('任务提交失败：' + e.message);
            cancelEcommerceFlow();
        }
    };

    const runEcommerceStep2 = async () => {
        ecFlowState.value = 'step2_loading';
        try {
            const res = await authFetch('/api/ecommerce/scenes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    items: ecStep1Data.value.map(item => ({
                        image_path: item.image_path,
                        image_name: item.image_name,
                        display_url: item.display_url,
                        description: item.description,
                    })),
                    scene_count: ecSettings.value.scene_count,
                }),
            });
            if (!res.ok) { const d = await res.json(); throw new Error(d.error || 'AI 场景生成失败'); }
            const data = await res.json();
            ecStep2Data.value = data.items;
            if (ecSettings.value.one_shot || ecSettings.value.skip_step2_review) {
                await runEcommerceJobSubmit();
            } else {
                ecFlowState.value = 'step2_review';
                setSubmitting(false);
            }
        } catch (e) {
            alert('场景方案生成失败：' + e.message);
            cancelEcommerceFlow();
        }
    };

    const proceedToStep2 = async () => {
        setSubmitting(true);
        await runEcommerceStep2();
    };

    const backToStep1 = () => { ecFlowState.value = 'step1_review'; };

    const proceedToSubmit = async () => {
        setSubmitting(true);
        await runEcommerceJobSubmit();
    };

    const startEcommerceFlow = async () => {
        if (form.value.files.length === 0) { alert('请上传至少一张产品图片'); return; }
        setSubmitting(true);
        ecFlowState.value = 'step1_loading';
        try {
            const fd = new FormData();
            form.value.files.forEach(f => fd.append('images', f));
            const res = await authFetch('/api/ecommerce/understand', { method: 'POST', body: fd });
            if (!res.ok) { const d = await res.json(); throw new Error(d.error || 'AI 产品分析失败'); }
            const data = await res.json();
            ecStep1Data.value = data.items;
            if (ecSettings.value.one_shot || ecSettings.value.skip_step1_review) {
                await runEcommerceStep2();
            } else {
                ecFlowState.value = 'step1_review';
                setSubmitting(false);
            }
        } catch (e) {
            alert('产品分析失败：' + e.message);
            cancelEcommerceFlow();
        }
    };

    const fetchSettings = async () => {
        try {
            const res = await authFetch('/api/settings');
            if (res.ok) {
                const data = await res.json();
                if (data.ecommerce) Object.assign(ecSettings.value, data.ecommerce);
            }
        } catch (e) {}
    };

    const saveEcSettings = async () => {
        try {
            await authFetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ecommerce: { ...ecSettings.value } }),
            });
        } catch (e) {}
    };

    return {
        ecFlowState, ecStep1Data, ecStep2Data, ecSettings,
        cancelEcommerceFlow, proceedToStep2, proceedToSubmit, backToStep1,
        startEcommerceFlow, fetchSettings, saveEcSettings, linkSubmitting,
    };
}
