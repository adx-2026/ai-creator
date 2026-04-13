/**
 * Template management composable: fetch, save, edit, delete prompt templates.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';

export function useTemplates(form) {
    const templates = ref({ public: [], private: [] });
    const showSaveTemplateModal = ref(false);
    const showTemplateManagerModal = ref(false);
    const newTemplate = ref({ name: '', content: '', negative_prompt: '', is_public: false });

    const fetchTemplates = async () => {
        try {
            const res = await authFetch('/api/templates');
            if (res.ok) templates.value = await res.json();
        } catch (e) { console.error(e); }
    };

    const handleTemplateSelect = () => {
        if (!form.value.selectedTemplate) {
            form.value.prompts = '';
            form.value.negative_prompt = '';
            return;
        }
        const [scope, name] = form.value.selectedTemplate.split(':');
        const list = scope === 'public' ? templates.value.public : templates.value.private;
        const t = list?.find(x => x.name === name);
        if (t) {
            form.value.prompts = t.content || '';
            form.value.negative_prompt = t.negative_prompt || '';
        }
    };

    const openSaveModal = () => {
        newTemplate.value = { name: '', content: '', negative_prompt: '', is_public: false };
        showSaveTemplateModal.value = true;
    };

    const openSaveModalFromResult = (res, job) => {
        newTemplate.value = {
            name: '',
            content: res.prompt || '',
            negative_prompt: job.negative_prompt || '',
            is_public: false,
        };
        showSaveTemplateModal.value = true;
    };

    const saveTemplate = async () => {
        if (!newTemplate.value.name || !newTemplate.value.content) return;
        try {
            const res = await authFetch('/api/templates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newTemplate.value),
            });
            if (res.ok) {
                fetchTemplates();
                form.value.prompts = newTemplate.value.content;
                form.value.negative_prompt = newTemplate.value.negative_prompt;
                form.value.selectedTemplate = (newTemplate.value.is_public ? 'public:' : 'private:') + newTemplate.value.name;
                showSaveTemplateModal.value = false;
            } else {
                const err = await res.json();
                alert(err.error || "保存失败");
            }
        } catch (e) { alert("网络异常导致保存失败"); }
    };

    const editTemplate = (scope, t) => {
        newTemplate.value = { name: t.name, content: t.content, negative_prompt: t.negative_prompt || '', is_public: scope === 'public' };
        showTemplateManagerModal.value = false;
        showSaveTemplateModal.value = true;
    };

    const deleteTemplate = async (scope, name) => {
        if (!confirm(`确定要删除此${scope === 'public' ? '公共' : '私有'}模板吗？`)) return;
        try {
            const res = await authFetch(`/api/templates/${scope}/${encodeURIComponent(name)}`, { method: 'DELETE' });
            if (res.ok) {
                fetchTemplates();
                if (form.value.selectedTemplate === `${scope}:${name}`) form.value.selectedTemplate = '';
            } else {
                const data = await res.json();
                alert(data.error || "删除失败");
            }
        } catch (e) { alert("删除请求发生错误！"); }
    };

    return {
        templates, showSaveTemplateModal, showTemplateManagerModal, newTemplate,
        fetchTemplates, handleTemplateSelect, openSaveModal, openSaveModalFromResult,
        saveTemplate, editTemplate, deleteTemplate,
    };
}
