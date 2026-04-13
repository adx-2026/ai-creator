/**
 * Authentication composable: login, logout, session check.
 */

import { ref } from 'vue';
import { authFetch, setUnauthorizedHandler } from '../api.js';

export function useAuth() {
    const user = ref(null);
    const isLoggingIn = ref(false);
    const loginForm = ref({ username: '', password: '' });

    setUnauthorizedHandler(() => {
        if (user.value) {
            user.value = null;
            localStorage.removeItem('auth_token');
            alert("登录已过期，请重新登录");
        }
    });

    const checkAuth = async (onSuccess) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
            try {
                const res = await authFetch('/api/me');
                if (res.ok) {
                    const data = await res.json();
                    user.value = { username: data.username, is_admin: data.is_admin };
                    if (onSuccess) onSuccess();
                } else {
                    localStorage.removeItem('auth_token');
                }
            } catch (e) {
                console.error("Auth check failed.", e);
            }
        }
    };

    const handleLogin = async (onSuccess) => {
        if (!loginForm.value.username || !loginForm.value.password) return;
        isLoggingIn.value = true;
        const fd = new FormData();
        fd.append("username", loginForm.value.username);
        fd.append("password", loginForm.value.password);
        try {
            const res = await fetch('/api/login', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok && data.access_token) {
                localStorage.setItem('auth_token', data.access_token);
                user.value = { username: data.username, is_admin: data.is_admin };
                loginForm.value = { username: '', password: '' };
                if (onSuccess) onSuccess();
            } else {
                alert(data.error || "登录失败");
            }
        } catch (e) {
            alert("网络错误");
        } finally {
            isLoggingIn.value = false;
        }
    };

    const handleLogout = async (onLogout) => {
        try { await authFetch('/api/logout', { method: 'POST' }); } catch (e) { }
        localStorage.removeItem('auth_token');
        user.value = null;
        if (onLogout) onLogout();
    };

    return { user, isLoggingIn, loginForm, checkAuth, handleLogin, handleLogout };
}
