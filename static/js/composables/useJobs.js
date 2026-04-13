/**
 * Job management composable: fetch, delete, retry jobs and subtasks.
 */

import { ref } from 'vue';
import { authFetch } from '../api.js';
import { MODE_LABELS } from '../constants.js';

export function useJobs(fetchCredit) {
    const jobs = ref([]);
    const isFetching = ref(false);
    const retryingJobs = ref({});
    const retryingTasks = ref({});
    const jobDetailModal = ref(null);

    const fetchJobs = async (user) => {
        if (!user) return;
        isFetching.value = true;
        try {
            const response = await authFetch('/api/jobs');
            if (response.ok) {
                jobs.value = await response.json();
                fetchCredit();
            }
        } catch (error) { console.error('Fetch jobs error:', error); }
        finally { isFetching.value = false; }
    };

    const deleteJob = async (id, user) => {
        if (!confirm('确定要彻底删除这条历史记录吗？')) return;
        try {
            const response = await authFetch('/api/jobs/' + id, { method: 'DELETE' });
            if (response.ok) fetchJobs(user);
            else alert('删除失败，权限不足或任务不存在。');
        } catch (error) { alert('删除请求发生错误！'); }
    };

    const hasFailedResults = (job) => {
        return !!job?.results?.some(res => res.status === 'error' && res.task_id);
    };

    const retryFailedTasks = async (job, user) => {
        retryingJobs.value[job.id] = true;
        try {
            const response = await authFetch(`/api/jobs/${job.id}/retry-failed`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '重试失败');
            await fetchJobs(user);
        } catch (error) {
            alert(error.message || '重试失败');
        } finally {
            delete retryingJobs.value[job.id];
        }
    };

    const retryTask = async (jobId, taskId, user) => {
        retryingTasks.value[taskId] = true;
        try {
            const response = await authFetch(`/api/jobs/${jobId}/retry-task/${taskId}`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '重试失败');
            await fetchJobs(user);
        } catch (error) {
            alert(error.message || '重试失败');
        } finally {
            delete retryingTasks.value[taskId];
        }
    };

    const getModeLabel = (mode) => MODE_LABELS[mode] || mode;

    const getSubtaskResult = (job, subtaskId) => (job.results || []).find(r => r.task_id === subtaskId) || null;

    const openJobDetail = (job) => { jobDetailModal.value = job; };

    const hasImages = (job) => {
        if (!job || !job.results) return false;
        return job.results.some(res =>
            (res.images && res.images.length > 0) || (res.videos && res.videos.length > 0)
        );
    };

    return {
        jobs, isFetching, retryingJobs, retryingTasks, jobDetailModal,
        fetchJobs, deleteJob, hasFailedResults, retryFailedTasks, retryTask,
        getModeLabel, getSubtaskResult, openJobDetail, hasImages,
    };
}
