import { createApp } from 'vue';
import App from './App.vue';

import '@/assets/styles/index.scss';

// 引入路由配置
import router from './router/index';

// 引入pinia
import { createPinia } from 'pinia';

// 引入ElementPlus
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';

// 将vue-i18n注入全局
import i18n from '@/utils/lang';

const app = createApp(App);
const pinia = createPinia();

// 注册
app.use(router);
app.use(pinia);
app.use(ElementPlus);
app.use(i18n);

app.mount('#app');
