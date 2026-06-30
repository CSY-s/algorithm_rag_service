<template>
  <div class="rag-chat-container" v-if="visible">
    <!-- 聊天窗口 -->
    <div class="rag-chat-window">
      <!-- 头部 -->
      <div class="chat-header">
        <div class="header-left">
          <svg class="icon" viewBox="0 0 1024 1024">
            <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64z m0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z" fill="currentColor"/>
            <path d="M623.6 316.7C593.6 290.4 554 276 512 276s-81.6 14.5-111.6 40.7C369.2 344 352 380.7 352 420v7.6c0 4.4 3.6 8 8 8h48c4.4 0 8-3.6 8-8V420c0-44.1 43.1-80 96-80s96 35.9 96 80c0 31.1-22 59.6-56.1 72.7-21.2 8.1-39.2 22.3-52.1 40.9-13.1 19-19.9 41.8-19.9 64.9V620c0 4.4 3.6 8 8 8h48c4.4 0 8-3.6 8-8v-22.7a48.3 48.3 0 0 1 30.9-44.8c59-22.7 97.1-74.7 97.1-132.5 0-39.3-17.2-76-48.4-103.3zM472 732a40 40 0 1 0 80 0 40 40 0 1 0-80 0z" fill="currentColor"/>
          </svg>
          <h3>智能助教</h3>
        </div>
        <button @click="close" class="close-btn" title="关闭">
          <svg viewBox="0 0 1024 1024">
            <path d="M563.8 512l262.5-312.9c4.4-5.2.7-13.1-6.1-13.1h-79.8c-4.7 0-9.2 2.1-12.3 5.7L511.6 449.8 295.1 191.7c-3-3.6-7.5-5.7-12.3-5.7H203c-6.8 0-10.5 7.9-6.1 13.1L459.4 512 196.9 824.9A7.95 7.95 0 0 0 203 838h79.8c4.7 0 9.2-2.1 12.3-5.7l216.5-258.1 216.5 258.1c3 3.6 7.5 5.7 12.3 5.7h79.8c6.8 0 10.5-7.9 6.1-13.1L563.8 512z" fill="currentColor"/>
          </svg>
        </button>
      </div>

      <!-- 消息列表 -->
      <div class="chat-messages" ref="messageList">
        <div v-if="messages.length === 0" class="empty-state">
          <svg class="empty-icon" viewBox="0 0 1024 1024">
            <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64z m0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z" fill="currentColor"/>
          </svg>
          <p>有问题尽管问我！</p>
        </div>

        <div 
          v-for="msg in messages" 
          :key="msg.id"
          :class="['message', msg.role]"
        >
          <div class="message-content">
            <div v-html="formatMessage(msg.content)"></div>
          </div>
          <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
        </div>
        
        <!-- 加载状态 -->
        <div v-if="loading" class="message assistant">
          <div class="message-content">
            <div class="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入框 -->
      <div class="chat-input">
        <textarea 
          v-model="inputText"
          @keydown.enter.exact.prevent="sendMessage"
          @keydown.shift.enter="insertNewLine"
          placeholder="输入你的问题... (Enter发送，Shift+Enter换行)"
          rows="1"
          ref="inputArea"
        ></textarea>
        <button 
          @click="sendMessage" 
          :disabled="loading || !inputText.trim()"
          class="send-btn"
          title="发送"
        >
          <svg viewBox="0 0 1024 1024">
            <path d="M931.4 498.9L94.9 79.5c-3.4-1.7-7.3-2.1-11-1.2-8.5 2.1-13.8 10.7-11.7 19.3l86.2 352.2c1.3 5.3 5.2 9.6 10.4 11.3l147.7 50.7-147.6 50.7c-5.2 1.8-9.1 6-10.3 11.3L72.2 926.5c-0.9 3.7-0.5 7.6 1.2 10.9 3.9 7.9 13.5 11.1 21.5 7.2l836.5-417c3.1-1.5 5.6-4.1 7.2-7.1 3.9-7.8 1.2-17.4-6.2-21.6z" fill="currentColor"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- 快捷操作按钮 -->
    <div class="quick-actions" v-if="!loading && messages.length > 0">
      <button @click="clearChat" class="quick-btn" title="新对话">
        <svg viewBox="0 0 1024 1024">
          <path d="M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6c-0.3 1.5-0.4 3-0.4 4.4 0 14.4 11.6 26 26 26h723c1.5 0 3-0.1 4.4-0.4 14.2-2.4 23.7-15.9 21.2-30zM204 390h272V182h72v208h272v104H204V390z m468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z" fill="currentColor"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'RAGChatWindow',
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    logicalContentId: {
      type: Number,
      required: true
    }
  },
  data() {
    return {
      messages: [],
      inputText: '',
      loading: false,
      sessionId: this.generateSessionId()
    }
  },
  watch: {
    visible(newVal) {
      if (newVal) {
        this.$nextTick(() => {
          this.scrollToBottom()
          this.$refs.inputArea?.focus()
        })
      }
    }
  },
  methods: {
    async sendMessage() {
      if (!this.inputText.trim() || this.loading) return
      
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: this.inputText,
        timestamp: new Date()
      }
      
      this.messages.push(userMessage)
      
      const question = this.inputText
      this.inputText = ''
      this.loading = true
      
      // 滚动到底部
      this.$nextTick(() => {
        this.scrollToBottom()
      })
      
      try {
        // 调用后端API
        const response = await axios.post('/api/rag/ask/', {
          question: question,
          logical_content_id: this.logicalContentId,
          session_id: this.sessionId
        })
        
        if (response.data.success) {
          const data = response.data.data
          
          // 添加AI回复
          const assistantMessage = {
            id: Date.now() + 1,
            role: 'assistant',
            content: data.answer,
            timestamp: new Date(),
            references: data.references
          }
          
          this.messages.push(assistantMessage)
          
          // 更新sessionId（如果服务端返回了新的）
          if (data.session_id) {
            this.sessionId = data.session_id
          }
        } else {
          throw new Error(response.data.error || '请求失败')
        }
        
      } catch (error) {
        console.error('RAG调用失败:', error)
        
        // 添加错误消息
        this.messages.push({
          id: Date.now() + 1,
          role: 'system',
          content: '抱歉，服务暂时不可用，请稍后再试。',
          timestamp: new Date()
        })
      } finally {
        this.loading = false
        
        // 滚动到底部
        this.$nextTick(() => {
          this.scrollToBottom()
          this.$refs.inputArea?.focus()
        })
      }
    },
    
    insertNewLine() {
      const textarea = this.$refs.inputArea
      const start = textarea.selectionStart
      const end = textarea.selectionEnd
      const text = this.inputText
      this.inputText = text.substring(0, start) + '\n' + text.substring(end)
      this.$nextTick(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 1
      })
    },
    
    formatMessage(text) {
      if (!text) return ''
      
      // 简单的Markdown格式化
      let formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>')
      
      // 处理chunk引用标记 [chunk_id=xx]
      formatted = formatted.replace(/\[chunk_id=(\d+)\]/g, '<sup class="chunk-ref">[$1]</sup>')
      
      return formatted
    },
    
    formatTime(date) {
      if (!date) return ''
      const d = new Date(date)
      return d.toLocaleTimeString('zh-CN', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    },
    
    generateSessionId() {
      return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    },
    
    scrollToBottom() {
      const container = this.$refs.messageList
      if (container) {
        container.scrollTop = container.scrollHeight
      }
    },
    
    async clearChat() {
      if (this.messages.length === 0) return
      
      if (!confirm('确定要开始新对话吗？当前对话记录将被清空。')) {
        return
      }
      
      try {
        // 调用后端清空会话
        await axios.delete(`/api/rag/session/${this.sessionId}/clear/`)
      } catch (error) {
        console.error('清空会话失败:', error)
      }
      
      // 清空本地消息
      this.messages = []
      
      // 生成新的sessionId
      this.sessionId = this.generateSessionId()
    },
    
    close() {
      this.$emit('close')
    }
  }
}
</script>

<style scoped>
.rag-chat-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.rag-chat-window {
  width: 400px;
  height: 600px;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-header {
  padding: 16px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-left .icon {
  width: 24px;
  height: 24px;
}

.chat-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: background 0.2s;
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.close-btn svg {
  width: 20px;
  height: 20px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f8f9fa;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
}

.empty-icon {
  width: 64px;
  height: 64px;
  margin-bottom: 16px;
  opacity: 0.3;
}

.message {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
}

.message.user {
  align-items: flex-end;
}

.message.assistant {
  align-items: flex-start;
}

.message-content {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  word-wrap: break-word;
  line-height: 1.5;
}

.message.user .message-content {
  background: #667eea;
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant .message-content {
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
  border-bottom-left-radius: 4px;
}

.message.system .message-content {
  background: #fff3cd;
  color: #856404;
  border: 1px solid #ffeaa7;
  max-width: 100%;
  text-align: center;
}

.message-content >>> code {
  background: rgba(0, 0, 0, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}

.message-content >>> .chunk-ref {
  color: #667eea;
  font-size: 0.8em;
  margin-left: 2px;
}

.message-time {
  font-size: 11px;
  color: #999;
  margin-top: 4px;
  padding: 0 4px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 4px 0;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #999;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-10px); }
}

.chat-input {
  padding: 16px 20px;
  border-top: 1px solid #e0e0e0;
  background: white;
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.chat-input textarea {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px 12px;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  max-height: 120px;
  transition: border-color 0.2s;
}

.chat-input textarea:focus {
  outline: none;
  border-color: #667eea;
}

.send-btn {
  padding: 10px 16px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  background: #5568d3;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.send-btn svg {
  width: 20px;
  height: 20px;
}

.quick-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.quick-btn {
  width: 40px;
  height: 40px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.quick-btn:hover {
  background: #f5f5f5;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.quick-btn svg {
  width: 20px;
  height: 20px;
  fill: #666;
}

/* 滚动条样式 */
.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: #ddd;
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: #ccc;
}
</style>
