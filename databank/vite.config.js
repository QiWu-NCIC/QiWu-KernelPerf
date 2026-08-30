import { defineConfig, loadEnv } from 'vite';
import path from 'path';
import vue from '@vitejs/plugin-vue';

import AutoImport from 'unplugin-auto-import/vite';

import Components from 'unplugin-vue-components/vite';
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers';

// https://vitejs.dev/config/
export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd()); // 获取.env文件里定义的环境变量

  return defineConfig({
    base: env.VITE_PUBLIC_PATH,
    plugins: [
      vue(),
      AutoImport({
        imports: ['vue', 'vue-router', 'vue-i18n'],
        resolvers: [ElementPlusResolver()],
      }),
      Components({
        //配置elementPlusResolver
        resolvers: [ElementPlusResolver({ importStyle: 'sass' })],
      }),
    ],
    build: {
      outDir: 'dist', //指定输出路径（相对于 项目根目录).
      assetsDir: 'assets', //指定生成静态资源的存放路径
    },
    server: {
      port: 8080,
      host: true,
      open: true,
      proxy: {
        // https://cn.vitejs.dev/config/#server-proxy
        // '/api': {
        //   target: 'http://mas.user.dev.cqxyy.net',
        //   changeOrigin: true,
        //   rewrite: (p) => p.replace(/^\/api/, '')
        // }
      },
    },
    resolve: {
      alias: {
        // 设置路径
        '~': path.resolve(__dirname, './'),
        // 设置别名
        '@': path.resolve(__dirname, 'src') + '/',
        'vue-i18n': 'vue-i18n/dist/vue-i18n.cjs.js',
      },
    },
    css: {
      // css预处理器
      preprocessorOptions: {
        scss: {
          api: 'modern-compiler', // or 'modern'
          // 全局变量文件
          additionalData: `
        @use "@/assets/styles/element.scss" as *;
        @use "@/assets/styles/var.scss" as *;
        `,
        },
      },
    },
  });
};
