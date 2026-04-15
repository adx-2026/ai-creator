/**
 * Form composable: form state, file handling, task calculation, job submission.
 */

import { ref, computed } from 'vue';
import { authFetch } from '../api.js';

export function useForm(form, currentTab, fetchJobs, fetchCredit, userValue, ecSettings, ecFlowState, tdFlowState, dimFlowState, startEcommerceFlow, startThreedFlow, startDimensionFlow) {

    const availableModels = ref([]);
    const isDragging = ref(false);
    const isSubmitting = ref(false);
    const fileInput = ref(null);
    const objectUrlCache = new WeakMap();

    const multiLineCount = computed(() =>
        form.value.prompts.split('\n').filter(l => l.trim()).length || 0
    );

    const isVideoFile = (file) => {
        return file.type.startsWith('video/') || /\.(mp4|mov)$/i.test(file.name);
    };

    const filteredModels = (group) => {
        const isVideoMode = ['video', 'multi_video'].includes(form.value.mode);
        return group.models.filter(m => isVideoMode ? m.type === 'video' : m.type !== 'video');
    };

    const calculateTotalTasks = () => {
        if (form.value.mode === 'video') return 1;
        if (['multi_t2i', 'multi_video'].includes(form.value.mode)) return multiLineCount.value || 1;
        if (form.value.mode === 'extract') return form.value.files.length > 0 ? form.value.files.length : 1;
        if (form.value.mode === 'ecommerce') return (form.value.files.length || 1) * (ecSettings.value.scene_count || 3);
        if (form.value.mode === 'threed') return form.value.files.length || 1;
        if (form.value.mode === 'dimension') return 1;
        let fileCount = ['i2i', 'fission', 'convert'].includes(form.value.mode) && form.value.files.length > 0 ? form.value.files.length : 1;
        let bSize = form.value.mode === 'fission' ? form.value.batch_size : 1;
        let ratioMul = form.value.mode === 'convert' ? Math.max(form.value.target_ratios.length, 1) : 1;
        return fileCount * bSize * ratioMul;
    };

    const isSubmitDisabled = computed(() => {
        if (isSubmitting.value || ecFlowState.value !== 'idle' || tdFlowState.value !== 'idle' || dimFlowState.value !== 'idle') return true;
        if (form.value.mode === 'convert') {
            if (form.value.files.length === 0 || form.value.target_ratios.length === 0) return true;
        } else if (['video', 'multi_video'].includes(form.value.mode)) {
            const needsRef = /r2v|i2v/.test(form.value.model_id);
            if (form.value.mode === 'video' && !form.value.prompts.trim()) return true;
            if (form.value.mode === 'multi_video' && multiLineCount.value === 0) return true;
            if (needsRef && form.value.files.length === 0) return true;
        } else if (form.value.mode === 'extract') {
            if (form.value.files.length === 0) return true;
        } else if (form.value.mode === 'ecommerce') {
            if (form.value.files.length === 0) return true;
        } else if (form.value.mode === 'threed') {
            if (form.value.files.length === 0) return true;
        } else if (form.value.mode === 'dimension') {
            if (form.value.files.length === 0) return true;
        } else if (form.value.mode === 'multi_t2i') {
            if (multiLineCount.value === 0) return true;
        } else {
            if (form.value.mode !== 'fission' && !form.value.prompts.trim()) return true;
            if (['i2i', 'fission'].includes(form.value.mode) && form.value.files.length === 0) return true;
        }
        return false;
    });

    const handlePasteAllowedModes = ['i2i', 'fission', 'convert', 'video', 'multi_video', 'extract', 'ecommerce', 'dimension'];

    // File handling
    const triggerFileInput = () => { if (fileInput.value) fileInput.value.click(); };
    const allowedImageTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'];
    const allowedVideoTypes = ['video/mp4', 'video/quicktime'];

    const isAllowedFile = (file) => {
        if (['video', 'multi_video'].includes(form.value.mode)) {
            return allowedImageTypes.includes(file.type) || allowedVideoTypes.includes(file.type) || /\.(mp4|mov)$/i.test(file.name);
        }
        return allowedImageTypes.includes(file.type);
    };

    const handleFileUpload = (event) => {
        if (event.target.files) Array.from(event.target.files).forEach(f => { if (isAllowedFile(f)) form.value.files.push(f); });
        event.target.value = '';
    };

    const onDrop = (event) => {
        isDragging.value = false;
        if (event.dataTransfer.files) Array.from(event.dataTransfer.files).forEach(f => { if (isAllowedFile(f)) form.value.files.push(f); });
    };

    const handlePaste = (event) => {
        const items = event.clipboardData && event.clipboardData.items;
        if (!items) return;
        if (!handlePasteAllowedModes.includes(form.value.mode)) return;
        Array.from(items).forEach(item => {
            if (item.kind === 'file') {
                const file = item.getAsFile();
                if (file && isAllowedFile(file)) form.value.files.push(file);
            }
        });
    };

    const removeFile = (index) => {
        const file = form.value.files[index];
        if (objectUrlCache.has(file)) { URL.revokeObjectURL(objectUrlCache.get(file)); objectUrlCache.delete(file); }
        form.value.files.splice(index, 1);
    };

    const removeAllFiles = () => {
        form.value.files.forEach(f => { if (objectUrlCache.has(f)) { URL.revokeObjectURL(objectUrlCache.get(f)); objectUrlCache.delete(f); } });
        form.value.files = [];
    };

    const getObjectURL = (file) => {
        if (!objectUrlCache.has(file)) objectUrlCache.set(file, URL.createObjectURL(file));
        return objectUrlCache.get(file);
    };

    // Model fetching
    const fetchModels = async () => {
        try {
            const res = await authFetch('/api/models');
            if (res.ok) {
                availableModels.value = await res.json();
                if (availableModels.value.length > 0 && availableModels.value[0].models.length > 0) {
                    let found = false;
                    availableModels.value.forEach(group => {
                        if (group.models.find(m => m.id === form.value.model_id)) found = true;
                    });
                    if (!found) form.value.model_id = availableModels.value[0].models[0].id;
                }
            }
        } catch (e) { console.error("Failed to load models"); }
    };

    // Job submission
    const buildVideoFormData = (prompt) => {
        const vp = form.value.videoParams;
        const fd = new FormData();
        fd.append('prompts', prompt);
        fd.append('negative_prompt', form.value.negative_prompt);
        fd.append('mode', 'video');
        fd.append('model_id', form.value.model_id);
        fd.append('batch_size', 1);
        fd.append('target_ratio', form.value.target_ratio);
        fd.append('video_size', vp.size);
        fd.append('video_duration', vp.duration);
        fd.append('video_shot_type', vp.shot_type);
        fd.append('video_audio', vp.audio);
        fd.append('video_watermark', vp.watermark);
        let tplName = form.value.selectedTemplate;
        if (tplName.includes(':')) tplName = tplName.split(':')[1];
        fd.append('template_name', tplName);
        form.value.files.forEach(file => fd.append('images', file));
        return fd;
    };

    const submitJob = async () => {
        if (form.value.mode === 'ecommerce') { await startEcommerceFlow(); return; }
        if (form.value.mode === 'threed') { await startThreedFlow(); return; }
        if (form.value.mode === 'dimension') { await startDimensionFlow(); return; }

        if (form.value.mode === 'multi_video') {
            const lines = form.value.prompts.split('\n').map(l => l.trim()).filter(l => l);
            if (lines.length === 0) return;
            isSubmitting.value = true;
            try {
                for (const line of lines) {
                    const fd = buildVideoFormData(line);
                    const resp = await authFetch('/api/jobs', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok || !data.id) { alert('提交失败: ' + (data.error || "未知原因")); break; }
                }
                form.value.prompts = ''; form.value.negative_prompt = ''; removeAllFiles();
                currentTab.value = 'jobs'; fetchJobs(userValue());
            } catch (error) { alert('请求发生错误: ' + error.message); }
            finally { isSubmitting.value = false; }
            return;
        }

        if (form.value.mode === 'convert' && form.value.target_ratios.length === 0) {
            alert("请至少选择一个目标尺寸比例"); return;
        }
        if (['i2i', 'fission', 'convert', 'extract'].includes(form.value.mode) && form.value.files.length === 0) {
            alert("请至少上传一张参考图片"); return;
        }
        if (form.value.mode !== 'convert' && form.value.mode !== 'fission' && form.value.mode !== 'extract' && !form.value.prompts.trim()) {
            return;
        }

        isSubmitting.value = true;
        let finalPrompt = form.value.prompts.trim();
        if (form.value.mode === 'convert' && !finalPrompt) {
            finalPrompt = "保持原图主体结构和风格不变，将画面自然延展或重绘以适应设定的新比例尺寸，边缘过渡自然。";
        }

        let tplName = form.value.selectedTemplate;
        if (tplName.includes(':')) tplName = tplName.split(':')[1];

        const buildFd = (ratio) => {
            const fd = new FormData();
            fd.append('prompts', finalPrompt);
            fd.append('negative_prompt', form.value.negative_prompt);
            fd.append('mode', form.value.mode);
            fd.append('model_id', form.value.model_id);
            fd.append('batch_size', ['fission', 'extract'].includes(form.value.mode) ? form.value.batch_size : 1);
            fd.append('target_ratio', ratio);
            if (form.value.mode === 'video') {
                const vp = form.value.videoParams;
                fd.append('video_size', vp.size);
                fd.append('video_duration', vp.duration);
                fd.append('video_shot_type', vp.shot_type);
                fd.append('video_audio', vp.audio);
                fd.append('video_watermark', vp.watermark);
            }
            fd.append('template_name', tplName);
            if (form.value.mode === 'fission') {
                fd.append('fission_title_template', form.value.fission_title_template || '');
            }
            if (['i2i', 'fission', 'convert', 'video', 'extract'].includes(form.value.mode)) {
                form.value.files.forEach(file => fd.append('images', file));
            }
            return fd;
        };

        const ratiosToSubmit = form.value.mode === 'convert'
            ? form.value.target_ratios.slice()
            : [form.value.target_ratio];

        try {
            let anyOk = false;
            for (const ratio of ratiosToSubmit) {
                const response = await authFetch('/api/jobs', { method: 'POST', body: buildFd(ratio) });
                const data = await response.json();
                if (response.ok && data.id) { anyOk = true; }
                else { alert('提交失败: ' + (data.error || "未知原因")); break; }
            }
            if (anyOk) {
                form.value.prompts = ''; form.value.negative_prompt = '';
                form.value.fission_title_template = '';
                removeAllFiles();
                currentTab.value = 'jobs'; fetchJobs(userValue());
            }
        } catch (error) { alert('请求发生错误: ' + error.message); }
        finally { isSubmitting.value = false; }
    };

    return {
        form, availableModels, isDragging, isSubmitting, fileInput, multiLineCount,
        isVideoFile, filteredModels, calculateTotalTasks, isSubmitDisabled,
        triggerFileInput, handleFileUpload, onDrop, handlePaste,
        removeFile, removeAllFiles, getObjectURL,
        fetchModels, submitJob,
    };
}
