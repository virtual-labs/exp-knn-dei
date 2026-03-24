// Service Worker - Minimal implementation for KNN Experiment
// This file is required by the build system

self.addEventListener('install', function(event) {
    self.skipWaiting();
});

self.addEventListener('activate', function(event) {
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', function(event) {
    event.respondWith(fetch(event.request));
});
