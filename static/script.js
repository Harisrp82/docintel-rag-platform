// RAG System Frontend JavaScript
class RAGSystem {
    constructor() { 
        this.messages = [];
        this.documents = [];
        this.chatSessions = [];
        this.currentSessionId = null;
        this.settings = {
            max_chunks: 3,
            similarity_threshold: 0.3,
            include_metadata: true
        };
    }
    
    init() {
        console.log('🚀 Initializing RAG System...');
        this.bindEvents();
        this.loadSettings();
        this.setupTheme();
        this.loadChatHistory();
        
        // Load documents with a small delay to ensure backend is ready
        setTimeout(() => {
            this.loadDocuments();
        }, 500);
        
        console.log('✅ RAG System initialized successfully');
    }
    
    bindEvents() {
        console.log('🔗 Binding events...');
        
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
            console.log('✅ Theme toggle bound');
        }
        
        // Settings
        const settingsBtn = document.getElementById('settings-btn');
        const closeSettings = document.getElementById('close-settings');
        const modalOverlay = document.getElementById('modal-overlay');
        
        if (settingsBtn) settingsBtn.addEventListener('click', () => this.openSettings());
        if (closeSettings) closeSettings.addEventListener('click', () => this.closeSettings());
        if (modalOverlay) modalOverlay.addEventListener('click', () => this.closeSettings());
        
        // Chat
        const sendBtn = document.getElementById('send-btn');
        const chatInput = document.getElementById('chat-input');
        
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
            console.log('✅ Send button bound');
        }
        
        if (chatInput) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            console.log('✅ Chat input bound');
        }
        
        // File upload
        const fileUpload = document.getElementById('file-upload');
        if (fileUpload) {
            fileUpload.addEventListener('change', (e) => this.handleFileUpload(e));
            console.log('✅ File upload bound');
        }
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // Chat history
        const clearHistoryBtn = document.getElementById('clear-history-btn');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => this.clearChatHistory());
        }
        
        // Export/Import
        const exportBtn = document.getElementById('export-btn');
        const importInput = document.getElementById('import-input');
        
        if (exportBtn) exportBtn.addEventListener('click', () => this.exportData());
        if (importInput) importInput.addEventListener('change', (e) => this.importData(e));
        
        // Settings sliders
        const maxChunks = document.getElementById('max-chunks');
        const similarityThreshold = document.getElementById('similarity-threshold');
        const includeMetadata = document.getElementById('include-metadata');
        
        if (maxChunks) {
            maxChunks.addEventListener('input', (e) => this.updateSetting('max_chunks', parseInt(e.target.value)));
        }
        if (similarityThreshold) {
            similarityThreshold.addEventListener('input', (e) => this.updateSetting('similarity_threshold', parseFloat(e.target.value) / 100));
        }
        if (includeMetadata) {
            includeMetadata.addEventListener('change', (e) => this.updateSetting('include_metadata', e.target.checked));
        }
        
        // Citation drawer
        const drawerOverlay = document.getElementById('drawer-overlay');
        const closeDrawer = document.getElementById('close-drawer');
        
        if (drawerOverlay) {
            drawerOverlay.addEventListener('click', () => this.closeCitation());
        }
        if (closeDrawer) {
            closeDrawer.addEventListener('click', () => this.closeCitation());
        }
        
        // Global event delegation for dynamically created buttons
        document.body.addEventListener('click', (e) => {
            const target = e.target.closest('.delete-doc-btn') || e.target.closest('[data-doc-id]') || e.target.closest('button[onclick*="removeDocument"]');
            if (target) {
                const docId = target.dataset.docId || target.getAttribute('data-doc-id') || (target.onclick && target.onclick.toString().match(/\d+/)[0]);
                if (docId) {
                    console.log('🗑️ Delete clicked for:', docId);
                    this.removeDocument(docId);
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
            }
        
            // Existing citation handler
            const citationTarget = e.target.closest('.citation-btn') || e.target.closest('[data-citation]') || e.target.closest('button[onclick*="showCitation"]');
            if (citationTarget) {
                const citation = citationTarget.dataset.citation || citationTarget.getAttribute('data-citation');
                const messageIndex = citationTarget.dataset.messageIndex || citationTarget.getAttribute('data-message-index');
                if (citation) {
                    console.log('📖 Citation clicked:', citation);
                    this.showCitation(citation, messageIndex);
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
            }
        
            // NEW: Edit message handler
            const editTarget = e.target.closest('.edit-btn');
            if (editTarget) {
                const messageIndex = editTarget.dataset.messageIndex;
                if (messageIndex !== undefined) {
                    console.log('✏️ Edit clicked for message:', messageIndex);
                    this.editMessage(parseInt(messageIndex));
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
            }
        
            // Existing session delete handler
            const sessionDelete = e.target.closest('.delete-session-btn') || e.target.closest('[data-session-id]');
            if (sessionDelete) {
                const sessionId = sessionDelete.dataset.sessionId || sessionDelete.getAttribute('data-session-id');
                if (sessionId) {
                    console.log('🗑️ Session delete clicked for:', sessionId);
                    this.deleteSession(sessionId);
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
            }
        });   
        console.log('🔗 All events bound');
    }
    
    setupTheme() {
        const savedTheme = localStorage.getItem('rag-theme') || 'dark';
        document.documentElement.classList.add(savedTheme);
        this.updateThemeIcon(savedTheme === 'dark');
    }
    
    toggleTheme() {
        const isDark = document.documentElement.classList.contains('dark');
        if (isDark) {
            document.documentElement.classList.remove('dark');
            document.documentElement.classList.add('light');
            localStorage.setItem('rag-theme', 'light');
        } else {
            document.documentElement.classList.remove('light');
            document.documentElement.classList.add('dark');
            localStorage.setItem('rag-theme', 'dark');
        }
        this.updateThemeIcon(!isDark);
    }
    
    updateThemeIcon(isDark) {
        const icon = document.querySelector('#theme-toggle i');
        if (icon) {
            icon.className = isDark ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
    
    switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.toggle('active', content.id === `${tabName}-tab`));
    }
    
    loadChatHistory() {
        const saved = localStorage.getItem('rag-chat-sessions');
        this.chatSessions = saved ? JSON.parse(saved) : [];
        this.renderChatHistory();
    }
    
    saveChatHistory() {
        localStorage.setItem('rag-chat-sessions', JSON.stringify(this.chatSessions));
    }
    
    createNewSession() {
        const sessionId = Date.now().toString();
        this.chatSessions.unshift({
            id: sessionId,
            title: 'New Chat',
            messages: [],
            timestamp: new Date().toISOString(),
            documentCount: 0
        });
        this.currentSessionId = sessionId;
        this.messages = [];
        this.saveChatHistory();
        this.renderChatHistory();
        this.renderMessages();
        return sessionId;
    }
    
    renderChatHistory() {
        const container = document.getElementById('chat-sessions-list');
        if (!container) return;
        
        if (this.chatSessions.length === 0) {
            container.innerHTML = '<div class="empty-state">No chat history yet. Start a conversation to see it here.</div>';
            return;
        }
        
        container.innerHTML = this.chatSessions.map(session => {
            const isActive = session.id === this.currentSessionId;
            const lastMessage = session.messages[session.messages.length - 1];
            const preview = lastMessage ? lastMessage.content.substring(0, 50) + '...' : 'No messages yet';
            return `
                <div class="chat-session-item ${isActive ? 'active' : ''}" data-session-id="${session.id}">
                    <div class="chat-session-header">
                        <div class="chat-session-title">${session.title}</div>
                        <div class="chat-session-meta">
                            <span>${session.messages.length} msgs</span>
                            <button class="btn btn-ghost btn-xs delete-session-btn" data-session-id="${session.id}">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="chat-session-preview">${preview}</div>
                </div>
            `;
        }).join('');
    }
    
    loadSession(sessionId) {
        const session = this.chatSessions.find(s => s.id === sessionId);
        if (!session) return;
        
        this.currentSessionId = sessionId;
        this.messages = session.messages;
        this.renderChatHistory();
        this.renderMessages();
        
        if (session.title === 'New Chat' && session.messages.length > 0) {
            const firstUserMessage = session.messages.find(m => m.role === 'user');
            if (firstUserMessage) {
                session.title = firstUserMessage.content.substring(0, 30) + '...';
                this.saveChatHistory();
                this.renderChatHistory();
            }
        }
    }
    
    deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this chat session?')) return;
        
        this.chatSessions = this.chatSessions.filter(s => s.id !== sessionId);
        if (this.currentSessionId === sessionId) {
            this.currentSessionId = null;
            this.messages = [];
            this.renderMessages();
        }
        this.saveChatHistory();
        this.renderChatHistory();
        this.showNotification('Chat session deleted', 'success');
    }
    
    clearChatHistory() {
        if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) return;
        
        this.chatSessions = [];
        this.currentSessionId = null;
        this.messages = [];
        this.saveChatHistory();
        this.renderChatHistory();
        this.renderMessages();
        this.showNotification('Chat history cleared', 'success');
    }
    
    async loadDocuments() {
        try {
            const response = await fetch('/documents');
            if (response.ok) {
                this.documents = await response.json();
                this.renderDocuments();
                this.updateChunkCount();
            } else {
                throw new Error('Failed to load documents');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Error loading documents', 'error');
        }
    }
    
    renderDocuments() {
        const container = document.getElementById('documents-list');
        if (!container) return;
        
        if (this.documents.length === 0) {
            container.innerHTML = '<div class="empty-state">No documents yet. Add one to get started.</div>';
            return;
        }
        
        container.innerHTML = this.documents.map(doc => `
            <div class="document-item" data-id="${doc.id}">
                <div class="document-info">
                    <i class="fas fa-file-alt"></i>
                    <div class="document-name">${doc.filename}</div>
                </div>
                <button class="btn btn-ghost btn-sm delete-doc-btn" data-doc-id="${doc.id}">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
    }
    
    updateChunkCount() {
        const chunkCountElement = document.getElementById('chunk-count');
        if (chunkCountElement) {
            const totalChunks = this.documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
            chunkCountElement.textContent = `${totalChunks} chunks`;
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/documents/upload', { method: 'POST', body: formData });
            if (response.ok) {
                await this.loadDocuments();
                this.showNotification('Document uploaded!', 'success');
            } else {
                throw new Error('Upload failed');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Upload failed', 'error');
        }
    }
    
    async removeDocument(docId) {
        if (!confirm('Are you sure?')) return;
        
        try {
            const response = await fetch(`/documents/${parseInt(docId)}`, { method: 'DELETE' });
            if (response.ok) {
                await this.loadDocuments();
                this.showNotification('Document removed', 'success');
            } else {
                throw new Error('Delete failed');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Delete failed', 'error');
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const question = input.value.trim();
        if (!question) return;
        input.value = '';
        
        if (!this.currentSessionId) this.createNewSession();
        
        this.addMessage('user', question);
        this.addMessage('assistant', 'Thinking...', [], true);
        
        try {
            console.log('Sending message:', question);
            console.log('Settings:', this.settings);
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question,
                    parameters: this.settings
                })
            });
            
            console.log('Response status:', response.status);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Response data:', data);
                this.updateLastMessage('assistant', data.answer, data.sources || []);
            } else {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                throw new Error(`Chat failed: ${response.status}`);
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.updateLastMessage('assistant', 'Error occurred: ' + error.message, []);
        }
    }
    
    addMessage(role, content, citations = [], isLoading = false) {
        this.messages.push({ role, content, citations, isLoading });
        if (this.currentSessionId) {
            const session = this.chatSessions.find(s => s.id === this.currentSessionId);
            if (session) {
                session.messages = this.messages;
                this.saveChatHistory();
                this.renderChatHistory();
            }
        }
        this.renderMessages();
        this.scrollToBottom();
    }
    
    updateLastMessage(role, content, citations) {
        const last = this.messages[this.messages.length - 1];
        if (last && last.role === role) {
            last.content = content;
            last.citations = citations;
            last.isLoading = false;
            if (this.currentSessionId) {
                const session = this.chatSessions.find(s => s.id === this.currentSessionId);
                if (session) {
                    session.messages = this.messages;
                    this.saveChatHistory();
                    this.renderChatHistory();
                }
            }
            this.renderMessages();
            this.scrollToBottom();
        }
    }
    
    renderMessages() {
        const container = document.getElementById('chat-messages');
        if (!container) return;
        
        if (this.messages.length === 0) {
            container.innerHTML = '<div class="empty-state">Ask a question to begin. Add your own docs in the Knowledge panel.</div>';
            return;
        }
        
        container.innerHTML = this.messages.map((msg, index) => `
            <div class="message ${msg.role}">
                <div class="message-content">
                    <div class="message-text">${msg.isLoading ? '<i class="fas fa-spinner fa-spin"></i> ' : ''}${msg.content}</div>
                    ${msg.role === 'user' ? `
                        <div class="message-actions">
                            <button class="edit-btn" data-message-index="${index}" title="Edit message">
                                <i class="fas fa-pen"></i>
                            </button>
                        </div>
                    ` : ''}
                    ${msg.citations && msg.citations.length > 0 ? `
                        <div class="citations">
                            <button class="citation-btn" data-citation="${msg.citations.join(', ')}" data-message-index="${index}">
                                <i class="fas fa-book"></i> ${msg.citations.length} sources
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
    
    showCitation(citation) {
        const drawer = document.getElementById('citation-drawer');
        const info = document.getElementById('drawer-info');
        const content = document.getElementById('drawer-content-text');
        
        if (drawer && info && content) {
            info.textContent = 'Sources';
            content.innerHTML = citation.split(', ').map(source => `<div class="citation-source"><i class="fas fa-file-alt"></i> ${source}</div>`).join('');
            drawer.classList.add('open');
        } else {
            alert('Sources: ' + citation);
        }
    }
    
    closeCitation() {
        const drawer = document.getElementById('citation-drawer');
        if (drawer) drawer.classList.remove('open');
    }
    
    scrollToBottom() {
        const container = document.getElementById('chat-messages');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }
    
    loadSettings() {
        const saved = localStorage.getItem('rag-settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }
    
    saveSettings() {
        localStorage.setItem('rag-settings', JSON.stringify(this.settings));
    }
    
    updateSetting(key, value) {
        this.settings[key] = value;
        this.saveSettings();
    }
    
    openSettings() {
        const modal = document.getElementById('settings-modal');
        if (modal) modal.classList.add('open');
    }
    
    closeSettings() {
        const modal = document.getElementById('settings-modal');
        if (modal) modal.classList.remove('open');
    }
    
    exportData() {
        const data = {
            chatSessions: this.chatSessions,
            settings: this.settings
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'rag-system-data.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    
    importData(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                if (data.chatSessions) this.chatSessions = data.chatSessions;
                if (data.settings) this.settings = data.settings;
                this.saveChatHistory();
                this.saveSettings();
                this.renderChatHistory();
                this.showNotification('Data imported successfully', 'success');
            } catch (error) {
                this.showNotification('Failed to import data', 'error');
            }
        };
        reader.readAsText(file);
    }
    
    showNotification(message, type) {
        console.log(`${type.toUpperCase()}: ${message}`);
        // You can add a proper notification system here if needed
    }

    editMessage(messageIndex) {
        const message = this.messages[messageIndex];
        if (!message || message.role !== 'user') return;
        
        const newContent = prompt('Edit your message:', message.content);
        if (newContent === null || newContent.trim() === '') return;
        
        const trimmedContent = newContent.trim();
        if (trimmedContent === message.content) return; // No change
        
        // Update the message
        message.content = trimmedContent;
        
        // Remove all messages after this one (since we're re-asking)
        this.messages = this.messages.slice(0, messageIndex + 1);
        
        // Update session
        if (this.currentSessionId) {
            const session = this.chatSessions.find(s => s.id === this.currentSessionId);
            if (session) {
                session.messages = this.messages;
                this.saveChatHistory();
                this.renderChatHistory();
            }
        }
        
        // Re-render messages
        this.renderMessages();
        
        // Send the edited message
        this.resendMessage(trimmedContent);
    }
    
    async resendMessage(question) {
        // Add loading message
        this.addMessage('assistant', 'Thinking...', [], true);
        
        try {
            console.log('Resending edited message:', question);
            console.log('Settings:', this.settings);
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question,
                    parameters: this.settings
                })
            });
            
            console.log('Response status:', response.status);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Response data:', data);
                this.updateLastMessage('assistant', data.answer, data.sources || []);
            } else {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                throw new Error(`Chat failed: ${response.status}`);
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.updateLastMessage('assistant', 'Error occurred: ' + error.message, []);
        }
    }
}

// Make ragSystem global
document.addEventListener('DOMContentLoaded', () => {
    window.ragSystem = new RAGSystem();
    ragSystem.init();
});