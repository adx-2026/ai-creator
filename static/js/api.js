/**
 * API layer: token management and authenticated fetch wrapper.
 */

let _onUnauthorized = null;

export function setUnauthorizedHandler(handler) {
    _onUnauthorized = handler;
}

export function getToken() {
    return localStorage.getItem('auth_token') || '';
}

export async function authFetch(url, options = {}) {
    const token = getToken();
    if (!options.headers) options.headers = {};
    if (token) options.headers['Authorization'] = 'Bearer ' + token;
    const res = await fetch(url, options);
    if (res.status === 401 && _onUnauthorized) {
        _onUnauthorized();
    }
    return res;
}
