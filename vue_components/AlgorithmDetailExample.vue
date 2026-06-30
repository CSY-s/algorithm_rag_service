<template>
  <div class="algorithm-detail-page">
    <!-- 现有的算法内容 -->
    <div class="algorithm-content">
      <h1>{{ algorithmName }}</h1>
      
      <!-- 算法步骤 -->
      <section class="section">
        <h2>算法步骤</h2>
        <div v-html="algorithmSteps"></div>
      </section>
      
      <!-- 代码实现 -->
      <section class="section">
        <h2>代码实现</h2>
        <pre><code>{{ algorithmCode }}</code></pre>
      </section>
      
      <!-- 复杂度分析 -->
      <section class="section">
        <h2>复杂度分析</h2>
        <div v-html="algorithmAnalysis"></div>
      </section>
    </div>
    
    <!-- 智能助教悬浮按钮 -->
    <button 
      @click="openRAGChat" 
      class="rag-trigger-btn"
      :class="{ 'has-badge': hasNewMessage }"
      title="智能助教"
    >
      <svg viewBox="0 0 1024 1024">
        <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64z m0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z" fill="currentColor"/>
        <path d="M623.6 316.7C593.6 290.4 554 276 512 276s-81.6 14.5-111.6 40.7C369.2 344 352 380.7 352 420v7.6c0 4.4 3.6 8 8 8h48c4.4 0 8-3.6 8-8V420c0-44.1 43.1-80 96-80s96 35.9 96 80c0 31.1-22 59.6-56.1 72.7-21.2 8.1-39.2 22.3-52.1 40.9-13.1 19-19.9 41.8-19.9 64.9V620c0 4.4 3.6 8 8 8h48c4.4 0 8-3.6 8-8v-22.7a48.3 48.3 0 0 1 30.9-44.8c59-22.7 97.1-74.7 97.1-132.5 0-39.3-17.2-76-48.4-103.3zM472 732a40 40 0 1 0 80 0 40 40 0 1 0-80 0z" fill="currentColor"/>
      </svg>
      <span class="btn-text">助教</span>
    </button>
    
    <!-- RAG聊天窗口 -->
    <RAGChatWindow 
      :visible="ragChatVisible"
      :logical-content-id="logicalContentId"
      @close="ragChatVisible = false"
    />
  </div>
</template>

<script>
import RAGChatWindow from './RAGChatWindow.vue'

export default {
  name: 'AlgorithmDetail',
  components: {
    RAGChatWindow
  },
  data() {
    return {
      logicalContentId: null,
      algorithmName: '',
      algorithmSteps: '',
      algorithmCode: '',
      algorithmAnalysis: '',
      ragChatVisible: false,
      hasNewMessage: false
    }
  },
  created() {
    // 从路由获取logicalContentId
    // 假设路由格式: /logicalContent/2
    this.logicalContentId = parseInt(this.$route.params.id)
    
    // 加载算法数据
    this.loadAlgorithmData()
  },
  methods: {
    async loadAlgorithmData() {
      try {
        // 调用你现有的API获取算法数据
        const response = await this.$http.get(`/api/logical-content/${this.logicalContentId}/`)
        
        const data = response.data
        this.algorithmName = data.name
        this.algorithmSteps = data.steps
        this.algorithmCode = data.code
        this.algorithmAnalysis = data.analysis
        
      } catch (error) {
        console.error('加载算法数据失败:', error)
        this.$message.error('加载失败')
      }
    },
    
    openRAGChat() {
      this.ragChatVisible = true
      this.hasNewMessage = false
    }
  }
}
</script>

<style scoped>
.algorithm-detail-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  position: relative;
}

.algorithm-content {
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.section {
  margin-bottom: 30px;
}

.section h2 {
  font-size: 20px;
  margin-bottom: 15px;
  color: #333;
  border-bottom: 2px solid #667eea;
  padding-bottom: 8px;
}

pre {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
}

code {
  font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.6;
}

/* 智能助教悬浮按钮 */
.rag-trigger-btn {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 50%;
  box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  transition: all 0.3s;
  z-index: 999;
}

.rag-trigger-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.rag-trigger-btn svg {
  width: 28px;
  height: 28px;
}

.btn-text {
  font-size: 11px;
  font-weight: 500;
}

.rag-trigger-btn.has-badge::after {
  content: '';
  position: absolute;
  top: 8px;
  right: 8px;
  width: 12px;
  height: 12px;
  background: #ff4444;
  border-radius: 50%;
  border: 2px solid white;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.1);
  }
}
</style>
