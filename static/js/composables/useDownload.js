/**
 * Download composable: file naming dialog and download helpers.
 */

import { ref, watch, nextTick } from 'vue';
import { getToken } from '../api.js';

export function useDownload() {
    const dlDialog = ref({ show: false, defaultName: '', inputName: '', resolve: null });

    const promptDownloadName = (defaultName) => new Promise(resolve => {
        dlDialog.value = { show: true, defaultName, inputName: defaultName, resolve };
    });

    const confirmDownloadName = () => {
        const name = dlDialog.value.inputName.trim() || dlDialog.value.defaultName;
        dlDialog.value.show = false;
        dlDialog.value.resolve(name);
    };

    const cancelDownload = () => {
        dlDialog.value.show = false;
        dlDialog.value.resolve(null);
    };

    watch(() => dlDialog.value.show, (val) => {
        if (val) nextTick(() => {
            const el = document.querySelector('.dl-name-input');
            if (el) { el.focus(); el.select(); }
        });
    });

    const downloadImage = async (url, filename) => {
        const name = await promptDownloadName(filename);
        if (name === null) return;
        try {
            const token = getToken();
            const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
            const response = await fetch(url, { headers });
            if (!response.ok) throw new Error("Network request failed");

            const blobUrl = URL.createObjectURL(await response.blob());
            const a = document.createElement('a');
            a.href = blobUrl; a.download = name; a.style.display = 'none';
            document.body.appendChild(a); a.click();
            setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(blobUrl); }, 1000);
        } catch (error) { alert("下载文件时遇到错误！"); }
    };

    const downloadAll = async (jobId) => {
        const defaultName = `job_${jobId.substring(0, 8)}_images.zip`;
        const name = await promptDownloadName(defaultName);
        if (name === null) return;
        try {
            const token = getToken();
            const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
            const response = await fetch(`/api/jobs/${jobId}/download`, { headers });
            if (!response.ok) throw new Error("下载压缩包失败");

            const blobUrl = URL.createObjectURL(await response.blob());
            const a = document.createElement('a');
            a.href = blobUrl; a.download = name; a.style.display = 'none';
            document.body.appendChild(a); a.click();
            setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(blobUrl); }, 1000);
        } catch (error) { alert("打包下载时遇到错误！"); }
    };

    return { dlDialog, confirmDownloadName, cancelDownload, downloadImage, downloadAll };
}
