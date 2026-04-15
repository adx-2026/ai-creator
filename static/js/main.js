/**
 * Root Vue application: wires all composables together and returns template bindings.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue';
import { getToken } from './api.js';
import { RATIO_OPTIONS } from './constants.js';

import { useAuth } from './composables/useAuth.js';
import { useCredits } from './composables/useCredits.js';
import { useTemplates } from './composables/useTemplates.js';
import { useJobs } from './composables/useJobs.js';
import { useDownload } from './composables/useDownload.js';
import { useForm } from './composables/useForm.js';
import { useEcommerce } from './composables/useEcommerce.js';
import { useThreed } from './composables/useThreed.js';
import { useDimension } from './composables/useDimension.js';

export default {
    setup() {
        const currentTab = ref('create');
        let pollInterval = null;

        // ── Auth ──
        const {
            user, isLoggingIn, loginForm,
            checkAuth, handleLogin: _handleLogin, handleLogout: _handleLogout,
        } = useAuth();

        const userValue = () => user.value;

        // ── Credits ──
        const {
            userCredit, isAliyunModel, fetchCredit,
            estimateCredits: _estimateCredits,
        } = useCredits();

        // ── Jobs ──
        const {
            jobs, isFetching, retryingJobs, retryingTasks, jobDetailModal,
            fetchJobs: _fetchJobs, deleteJob: _deleteJob,
            hasFailedResults, retryFailedTasks: _retryFailedTasks, retryTask: _retryTask,
            getModeLabel, getSubtaskResult, openJobDetail, hasImages,
        } = useJobs(fetchCredit);

        // Wrap job functions to auto-pass user (template calls without user arg)
        const fetchJobs = () => _fetchJobs(user.value);
        const deleteJob = (id) => _deleteJob(id, user.value);
        const retryFailedTasks = (job) => _retryFailedTasks(job, user.value);
        const retryTask = (jobId, taskId) => _retryTask(jobId, taskId, user.value);

        // ── Download ──
        const { dlDialog, confirmDownloadName, cancelDownload, downloadImage, downloadAll } = useDownload();

        // ── Shared form ref (created here to resolve circular dependency) ──
        const form = ref({
            mode: 't2i',
            model_id: 'gemini-3.1-flash-image',
            prompts: '',
            negative_prompt: '',
            files: [],
            selectedTemplate: '',
            batch_size: 4,
            target_ratio: '1:1',
            target_ratios: [],
            videoParams: { size: '1280*720', duration: 5, shot_type: 'single', audio: true, watermark: false },
            fission_title_template: '',
        });

        // ── Ecommerce flow ──
        const {
            ecFlowState, ecStep1Data, ecStep2Data, ecSettings,
            cancelEcommerceFlow, proceedToStep2, proceedToSubmit, backToStep1,
            startEcommerceFlow, fetchSettings, saveEcSettings, linkSubmitting: ecLinkSubmitting,
        } = useEcommerce(form, currentTab, fetchJobs, fetchCredit, userValue);

        // ── 3D conversion flow ──
        const {
            tdFlowState, tdStep1Data, tdSettings,
            cancelThreedFlow, proceedToThreedSubmit,
            startThreedFlow, linkSubmitting: tdLinkSubmitting,
        } = useThreed(form, currentTab, fetchJobs, fetchCredit, userValue);

        // ── Dimension annotation flow ──
        const {
            dimState, dimResult, dimError, dimSettings,
            cancelDimensionFlow, startDimensionFlow, linkSubmitting: dimLinkSubmitting,
        } = useDimension(form, fetchCredit);

        // ── Form & submission ──
        const {
            availableModels, isDragging, isSubmitting, fileInput, multiLineCount,
            isVideoFile, filteredModels, calculateTotalTasks,
            isSubmitDisabled: _isSubmitDisabled,
            triggerFileInput, handleFileUpload, onDrop, handlePaste,
            removeFile, removeAllFiles, getObjectURL,
            fetchModels, submitJob: _submitJob,
        } = useForm(
            form, currentTab, fetchJobs, fetchCredit, userValue,
            ecSettings, ecFlowState, tdFlowState, dimState,
            startEcommerceFlow, startThreedFlow, startDimensionFlow,
        );

        // Link ecommerce/threed/dimension submitting state to form's isSubmitting
        ecLinkSubmitting((v) => { isSubmitting.value = v; });
        tdLinkSubmitting((v) => { isSubmitting.value = v; });
        dimLinkSubmitting((v) => { isSubmitting.value = v; });

        // ── Templates ──
        const {
            templates, showSaveTemplateModal, showTemplateManagerModal, newTemplate,
            fetchTemplates, handleTemplateSelect, openSaveModal, openSaveModalFromResult,
            saveTemplate, editTemplate, deleteTemplate,
        } = useTemplates(form);

        // ── Credit-aware estimateCredits wrapper (template calls with no args) ──
        const estimateCredits = () =>
            _estimateCredits(form.value, calculateTotalTasks, ecSettings.value, multiLineCount.value);

        // ── Extended isSubmitDisabled (adds credit balance check) ──
        const isSubmitDisabled = computed(() => {
            if (_isSubmitDisabled.value) return true;
            if (isAliyunModel(form.value.model_id) && userCredit.value !== null && userCredit.value < estimateCredits()) return true;
            return false;
        });

        // ── Standalone utilities ──
        const formatDate = (timestamp) => {
            if (!timestamp) return '';
            return new Date(timestamp * 1000).toLocaleString();
        };

        const renderMarkdown = (text) => text ? marked.parse(text) : '';

        const copyToClipboard = async (text) => {
            try {
                await navigator.clipboard.writeText(text);
            } catch {
                const el = document.createElement('textarea');
                el.value = text;
                el.style.position = 'fixed';
                el.style.opacity = '0';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
            }
        };

        // ── Lifecycle orchestration ──
        const initializeUserSpace = () => {
            fetchModels();
            fetchTemplates();
            fetchJobs();
            fetchCredit();
            fetchSettings();
            if (!pollInterval) {
                pollInterval = setInterval(() => {
                    if (user.value) {
                        fetchCredit();
                        if (currentTab.value === 'jobs') fetchJobs();
                    }
                }, 3000);
            }
        };

        const handleLogin = () => _handleLogin(initializeUserSpace);

        const handleLogout = () => _handleLogout(() => {
            userCredit.value = null;
            jobs.value = [];
            templates.value = { public: [], private: [] };
        });

        onMounted(() => {
            checkAuth(initializeUserSpace);
            document.addEventListener('paste', handlePaste);
        });

        onUnmounted(() => {
            if (pollInterval) clearInterval(pollInterval);
            removeAllFiles();
            document.removeEventListener('paste', handlePaste);
        });

        // ── Return all template bindings ──
        return {
            // Constants
            RATIO_OPTIONS,

            // Auth
            user, isLoggingIn, loginForm, handleLogin, handleLogout,

            // Navigation
            currentTab,

            // Credits
            userCredit, isAliyunModel, estimateCredits,

            // Form & file handling
            form, availableModels, isDragging, isSubmitting, fileInput, multiLineCount,
            isVideoFile, filteredModels, calculateTotalTasks, isSubmitDisabled,
            triggerFileInput, handleFileUpload, onDrop, handlePaste,
            removeFile, removeAllFiles, getObjectURL,
            submitJob: _submitJob,

            // Templates
            templates, showSaveTemplateModal, showTemplateManagerModal, newTemplate,
            handleTemplateSelect, openSaveModal, openSaveModalFromResult,
            saveTemplate, editTemplate, deleteTemplate,

            // Jobs
            jobs, isFetching, retryingJobs, retryingTasks, jobDetailModal,
            fetchJobs, deleteJob, hasFailedResults, retryFailedTasks, retryTask,
            getModeLabel, getSubtaskResult, openJobDetail, hasImages,

            // Download
            dlDialog, confirmDownloadName, cancelDownload, downloadImage, downloadAll,

            // Ecommerce flow
            ecFlowState, ecStep1Data, ecStep2Data, ecSettings,
            cancelEcommerceFlow, proceedToStep2, proceedToSubmit, backToStep1, saveEcSettings,

            // 3D flow
            tdFlowState, tdStep1Data, tdSettings,
            cancelThreedFlow, proceedToThreedSubmit,

            // Dimension annotation flow
            dimState, dimResult, dimError, dimSettings, cancelDimensionFlow,

            // Utilities
            formatDate, renderMarkdown, getToken, copyToClipboard,
        };
    },
};
